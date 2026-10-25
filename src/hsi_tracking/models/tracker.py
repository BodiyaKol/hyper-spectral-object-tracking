"""HSI adapter composition and one-pass tracking interface for ViPT."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from hsi_tracking.models.prompt_adapter import CNNPromptAdapter
from hsi_tracking.models.spectral_projector import SpectralProjector

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VIPT_ROOT = PROJECT_ROOT / "third_party" / "ViPT"


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve auto/cpu/cuda and fail clearly for unavailable requested CUDA."""
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch runtime has no available CUDA device"
        )
    return resolved


class HSIAdaptedViPT(nn.Module):
    """Frozen pretrained ViPT with a trainable projector and CNN prompt adapter."""

    def __init__(self, vipt_model: nn.Module, sensor: str, input_scale: float = 255.0) -> None:
        super().__init__()
        if input_scale <= 0:
            raise ValueError("input_scale must be positive")

        self.vipt_model = vipt_model
        self.sensor = sensor
        self.input_scale = float(input_scale)
        transformer = self.vipt_model.backbone
        self.patch_stride = int(transformer.patch_size)
        self.embed_dim = int(transformer.embed_dim)
        self.num_layers = len(transformer.blocks)
        self.projector = SpectralProjector()
        self.prompt_adapter = CNNPromptAdapter(
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
        )
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        for parameter in self.vipt_model.parameters():
            parameter.requires_grad = False
        self.vipt_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.vipt_model.eval()
        return self

    def trainable_parameters(self):
        return list(self.projector.parameters()) + list(self.prompt_adapter.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters())

    def _prepare_stream(self, cube: Tensor):
        scaled_cube = cube.float() / self.input_scale
        pseudo_rgb = self.projector(scaled_cube, self.sensor)
        pseudo_rgb = (pseudo_rgb - self.imagenet_mean) / self.imagenet_std
        grid_size = (
            pseudo_rgb.shape[-2] // self.patch_stride,
            pseudo_rgb.shape[-1] // self.patch_stride,
        )
        if min(grid_size) < 1:
            raise ValueError(
                f"HSI crop {tuple(pseudo_rgb.shape[-2:])} is smaller than ViPT patch stride"
            )
        prompts = self.prompt_adapter(scaled_cube, self.sensor, grid_size)
        return pseudo_rgb, prompts

    def forward(self, template_hsi: Tensor, search_hsi: Tensor):
        """Run two HSI crops and return ViPT's box-head output dictionary."""
        template_rgb, template_prompts = self._prepare_stream(template_hsi)
        search_rgb, search_prompts = self._prepare_stream(search_hsi)
        external_prompts = [
            torch.cat((template_prompt, search_prompt), dim=1)
            for template_prompt, search_prompt in zip(template_prompts, search_prompts)
        ]
        return self.vipt_model(
            template=template_rgb,
            search=search_rgb,
            external_prompts=external_prompts,
        )

    def save_adapters(self, path: str | Path, extra: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sensor": self.sensor,
                "input_scale": self.input_scale,
                "patch_stride": self.patch_stride,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "projector": self.projector.state_dict(),
                "prompt_adapter": self.prompt_adapter.state_dict(),
                "extra": extra or {},
            },
            str(path),
        )
        return path

    def load_adapters(self, path: str | Path) -> dict:
        state = torch.load(str(path), map_location="cpu")
        if state["sensor"] != self.sensor:
            raise ValueError(
                f"Checkpoint sensor {state['sensor']} does not match model sensor {self.sensor}"
            )
        self.projector.load_state_dict(state["projector"])
        self.prompt_adapter.load_state_dict(state["prompt_adapter"])
        self.input_scale = float(state.get("input_scale", self.input_scale))
        return state


def build_hsi_adapted_vipt(
    sensor: str,
    checkpoint: str | Path,
    device: str | torch.device = "auto",
    vipt_root: str | Path = DEFAULT_VIPT_ROOT,
    input_scale: float = 255.0,
):
    """Build patched ViPT from its pretrained RGB-T checkpoint and attach HSI adapters."""
    vipt_root = Path(vipt_root).expanduser().resolve()
    if not (vipt_root / ".git").exists():
        raise FileNotFoundError(f"ViPT checkout not found: {vipt_root}")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ViPT checkpoint not found: {checkpoint}")
    if str(vipt_root) not in sys.path:
        sys.path.insert(0, str(vipt_root))
    device = resolve_device(device)

    from lib.config.vipt.config import cfg, update_config_from_file
    from lib.models.vipt import build_viptrack

    config_path = vipt_root / "experiments" / "vipt" / "deep_rgbt.yaml"
    update_config_from_file(str(config_path))
    cfg.MODEL.BACKBONE.CE_LOC = []
    cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []
    vipt_model = build_viptrack(cfg, training=False)
    pretrained = torch.load(str(checkpoint), map_location="cpu")
    vipt_model.load_state_dict(pretrained["net"], strict=True)
    model = HSIAdaptedViPT(vipt_model, sensor=sensor, input_scale=input_scale).to(device)
    return model, cfg


class HSISequenceTracker:
    """Initialize with one HSI frame/BBox, then track HSI frames in HWC format."""

    def __init__(self, model: HSIAdaptedViPT, cfg, device: str | torch.device = "auto") -> None:
        if str(DEFAULT_VIPT_ROOT) not in sys.path:
            sys.path.insert(0, str(DEFAULT_VIPT_ROOT))
        from lib.test.utils.hann import hann2d
        from lib.train.data.processing_utils import sample_target
        from lib.utils.box_ops import clip_box

        self.model = model
        self.cfg = cfg
        self.device = resolve_device(device)
        self.sample_target = sample_target
        self.clip_box = clip_box
        self.template_factor = cfg.TEST.TEMPLATE_FACTOR
        self.template_size = cfg.TEST.TEMPLATE_SIZE
        self.search_factor = cfg.TEST.SEARCH_FACTOR
        self.search_size = cfg.TEST.SEARCH_SIZE
        self.feat_size = self.search_size // model.patch_stride
        self.output_window = hann2d(
            torch.tensor([self.feat_size, self.feat_size]).long(), centered=True
        ).to(self.device)
        self.state = None

    @staticmethod
    def _as_tensor(crop: np.ndarray, device: torch.device) -> Tensor:
        chw = np.ascontiguousarray(crop.transpose(2, 0, 1))
        return torch.from_numpy(chw).unsqueeze(0).to(device=device, dtype=torch.float32)

    def initialize(self, hsi_hwc: np.ndarray, initial_bbox) -> None:
        self.height, self.width = hsi_hwc.shape[:2]
        self.state = [float(value) for value in initial_bbox]
        template_crop, _, _ = self.sample_target(
            hsi_hwc, self.state, self.template_factor, output_sz=self.template_size
        )
        self.template = self._as_tensor(template_crop, self.device)

    def _map_box_back(self, pred_box, resize_factor):
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, width, height = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + cx_prev - half_side
        cy_real = cy + cy_prev - half_side
        return [cx_real - width / 2, cy_real - height / 2, width, height]

    def track(self, hsi_hwc: np.ndarray) -> dict:
        if self.state is None:
            raise RuntimeError("Call initialize() before track()")
        search_crop, resize_factor, _ = self.sample_target(
            hsi_hwc, self.state, self.search_factor, output_sz=self.search_size
        )
        search = self._as_tensor(search_crop, self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.template, search)
            response = self.output_window * output["score_map"]
            boxes, scores = self.model.vipt_model.box_head.cal_bbox(
                response, output["size_map"], output["offset_map"], return_score=True
            )
        pred_box = (boxes.view(-1, 4).mean(dim=0) * self.search_size / resize_factor).tolist()
        self.state = self.clip_box(
            self._map_box_back(pred_box, resize_factor), self.height, self.width, margin=10
        )
        return {"target_bbox": self.state, "best_score": float(scores[0][0].item())}
