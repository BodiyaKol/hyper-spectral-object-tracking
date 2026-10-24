"""Minimal pairwise training loop for the HSI projector and CNN prompt adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from hsi_tracking.data.dataset import HSI_EXTENSIONS, load_hsi_sequence
from hsi_tracking.models.tracker import build_hsi_adapted_vipt, resolve_device


class HSIFramePairDataset(Dataset):
    """Adjacent labeled frame pairs for template/search training."""

    def __init__(
        self, sequence_dirs, sensor, template_size, search_size, template_factor, search_factor
    ):
        from lib.train.data.processing_utils import sample_target, transform_image_to_crop

        self.sample_target = sample_target
        self.transform_image_to_crop = transform_image_to_crop
        self.sensor = sensor
        self.template_size = template_size
        self.search_size = search_size
        self.template_factor = template_factor
        self.search_factor = search_factor
        self.sequences = []
        self.pairs = []

        for sequence_dir in sequence_dirs:
            sequence = load_hsi_sequence(sequence_dir, sensor)
            if sequence.annotation_kind != "groundtruth":
                raise ValueError(f"Training needs groundtruth_rect.txt: {sequence_dir}")
            sequence_index = len(self.sequences)
            self.sequences.append(sequence)
            valid = [self._valid_box(box) for box in sequence.boxes]
            self.pairs.extend(
                (sequence_index, frame_index, frame_index + 1)
                for frame_index in range(len(sequence) - 1)
                if valid[frame_index] and valid[frame_index + 1]
            )

        if not self.pairs:
            raise ValueError("No adjacent frame pairs with valid ground-truth boxes were found")

    @staticmethod
    def _valid_box(box):
        return box is not None and box[0] >= 0 and box[1] >= 0 and box[2] > 0 and box[3] > 0

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _to_chw(crop):
        return torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1))).float()

    def __getitem__(self, pair_index):
        sequence_index, template_index, search_index = self.pairs[pair_index]
        sequence = self.sequences[sequence_index]
        template_cube = sequence.load_cube(template_index)
        search_cube = sequence.load_cube(search_index)
        template_box = list(sequence.boxes[template_index])
        search_box = torch.tensor(sequence.boxes[search_index], dtype=torch.float32)

        template_crop, _, _ = self.sample_target(
            template_cube,
            template_box,
            self.template_factor,
            output_sz=self.template_size,
        )
        search_crop, resize_factor, _ = self.sample_target(
            search_cube,
            template_box,
            self.search_factor,
            output_sz=self.search_size,
        )
        crop_box = torch.tensor(template_box, dtype=torch.float32)
        target_xywh = self.transform_image_to_crop(
            search_box,
            crop_box,
            resize_factor,
            torch.tensor([self.search_size, self.search_size], dtype=torch.float32),
            normalize=True,
        )
        x, y, width, height = target_xywh.unbind()
        target_cxcywh = torch.stack((x + width / 2, y + height / 2, width, height))

        return self._to_chw(template_crop), self._to_chw(search_crop), target_cxcywh


def discover_training_sequences(data_root):
    """Accept either one sequence folder or a root containing sequence folders."""
    data_root = Path(data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Training data folder does not exist: {data_root}")
    if (data_root / "groundtruth_rect.txt").is_file() and any(
        path.suffix.lower() in HSI_EXTENSIONS for path in data_root.iterdir()
    ):
        return [data_root]
    sequences = sorted(
        path.parent
        for path in data_root.rglob("groundtruth_rect.txt")
        if any(frame.suffix.lower() in HSI_EXTENSIONS for frame in path.parent.iterdir())
    )
    if not sequences:
        raise ValueError(
            f"No HSI sequence folders with groundtruth_rect.txt found under {data_root}"
        )
    return sequences


def _cxcywh_to_xyxy(boxes):
    center = boxes[..., :2]
    half_size = boxes[..., 2:] / 2
    return torch.cat((center - half_size, center + half_size), dim=-1)


def _giou_loss(prediction, target):
    pred_xyxy = _cxcywh_to_xyxy(prediction)
    target_xyxy = _cxcywh_to_xyxy(target)
    intersection_lt = torch.maximum(pred_xyxy[..., :2], target_xyxy[..., :2])
    intersection_rb = torch.minimum(pred_xyxy[..., 2:], target_xyxy[..., 2:])
    intersection_wh = (intersection_rb - intersection_lt).clamp(min=0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

    pred_wh = (pred_xyxy[..., 2:] - pred_xyxy[..., :2]).clamp(min=0)
    target_wh = (target_xyxy[..., 2:] - target_xyxy[..., :2]).clamp(min=0)
    union = pred_wh[..., 0] * pred_wh[..., 1] + target_wh[..., 0] * target_wh[..., 1] - intersection
    iou = intersection / union.clamp(min=1e-6)

    enclosing_lt = torch.minimum(pred_xyxy[..., :2], target_xyxy[..., :2])
    enclosing_rb = torch.maximum(pred_xyxy[..., 2:], target_xyxy[..., 2:])
    enclosing_wh = (enclosing_rb - enclosing_lt).clamp(min=0)
    enclosing = enclosing_wh[..., 0] * enclosing_wh[..., 1]
    giou = iou - (enclosing - union) / enclosing.clamp(min=1e-6)
    return 1 - giou


def train_hsi_adapters(
    data_root,
    sensor,
    base_checkpoint,
    output_dir,
    epochs=1,
    batch_size=1,
    learning_rate=1e-4,
    giou_weight=2.0,
    max_steps=0,
    max_sequences=None,
    input_scale=255.0,
    num_workers=0,
    device="auto",
):
    """Train only the spectral projector and CNN prompt adapter; return history."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")

    device = resolve_device(device)
    sequence_dirs = discover_training_sequences(data_root)
    if max_sequences is not None:
        if max_sequences < 1:
            raise ValueError("max_sequences must be positive or None")
        sequence_dirs = sequence_dirs[:max_sequences]
        if not sequence_dirs:
            raise ValueError("No training sequences selected")
    model, cfg = build_hsi_adapted_vipt(
        sensor=sensor,
        checkpoint=base_checkpoint,
        device=device,
        input_scale=input_scale,
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = model.trainable_parameter_count()
    trainable_ratio = trainable_parameters / float(total_parameters)
    if trainable_ratio >= 0.03:
        raise ValueError(
            f"Trainable parameter budget is {trainable_ratio:.2%}; proposal limit is below 3%"
        )
    print(
        f"Sensor={sensor} sequences={len(sequence_dirs)} trainable={trainable_parameters:,}/{total_parameters:,} ({trainable_ratio:.3%})"
    )
    model.train()
    dataset = HSIFramePairDataset(
        sequence_dirs,
        sensor,
        cfg.DATA.TEMPLATE.SIZE,
        cfg.DATA.SEARCH.SIZE,
        cfg.DATA.TEMPLATE.FACTOR,
        cfg.DATA.SEARCH.FACTOR,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    print(
        f"Training pairs={len(dataset)} batch_size={batch_size} optimizer=AdamW lr={learning_rate}"
    )
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=learning_rate)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    global_step = 0

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for template, search, target in loader:
            template = template.to(device)
            search = search.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            outputs = model(template, search)
            prediction = outputs["pred_boxes"].squeeze(1)
            l1 = F.smooth_l1_loss(prediction, target)
            giou = _giou_loss(prediction, target).mean()
            loss = l1 + giou_weight * giou
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=0.1)
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))
            global_step += 1
            if global_step % 10 == 0 or global_step == 1:
                print(
                    f"epoch {epoch + 1}/{epochs} step {global_step}/{len(loader) * epochs} loss={float(loss.detach()):.4f} l1={float(l1.detach()):.4f} giou={float(giou.detach()):.4f}"
                )
            if max_steps and global_step >= max_steps:
                break

        record = {
            "epoch": epoch + 1,
            "steps": len(epoch_losses),
            "mean_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
            "trainable_parameters": trainable_parameters,
            "trainable_parameter_ratio": trainable_ratio,
        }
        history.append(record)
        model.save_adapters(
            output_dir / "adapter_latest.pth",
            extra={"sensor": sensor, "epoch": epoch + 1, "history": history},
        )
        print("Saved adapter checkpoint; epoch summary:", record)
        if max_steps and global_step >= max_steps:
            break

    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    return history
