"""CNN adapter that converts full HSI cubes into ViPT layer prompts."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F

from hsi_tracking.data.x2cube import SENSORS, get_sensor_spec


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, features: Tensor) -> Tensor:
        return F.gelu(features + self.layers(features))


class CNNPromptAdapter(nn.Module):
    """Generate one spatial prompt-token tensor per transformer layer.

    Input is an HSI crop in ``B x C x H x W`` format. Output is a list of
    ``B x (grid_h * grid_w) x embed_dim`` tensors, ready to be injected into
    the matching ViPT transformer layers. The spectral stem is sensor-specific;
    later spatial blocks and per-layer heads are shared.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_layers: int = 12,
        hidden_channels: int = 48,
        norm_groups: int = 8,
        spatial_blocks: int = 2,
    ) -> None:
        super().__init__()
        if hidden_channels % norm_groups:
            raise ValueError("hidden_channels must be divisible by norm_groups")
        if num_layers < 1 or spatial_blocks < 1:
            raise ValueError("num_layers and spatial_blocks must be positive")

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.stems = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Conv2d(spec.expected_bands, hidden_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(norm_groups, hidden_channels),
                    nn.GELU(),
                )
                for name, spec in SENSORS.items()
            }
        )
        self.spatial_blocks = nn.Sequential(
            *(_ResidualConvBlock(hidden_channels, norm_groups) for _ in range(spatial_blocks))
        )
        self.prompt_heads = nn.ModuleList(
            nn.Conv2d(hidden_channels, embed_dim, kernel_size=1) for _ in range(num_layers)
        )

    def forward(
        self,
        cube: Tensor,
        sensor: str,
        output_size: tuple[int, int],
    ) -> list[Tensor]:
        """Return layer prompts shaped ``B x N x embed_dim`` for ``N`` grid cells."""
        spec = get_sensor_spec(sensor)
        if cube.ndim != 4:
            raise ValueError(f"Expected HSI tensor in BCHW format, got shape {tuple(cube.shape)}")
        if cube.shape[1] != spec.expected_bands:
            raise ValueError(
                f"{spec.name} prompt adapter expects {spec.expected_bands} channels, "
                f"got {cube.shape[1]}"
            )
        if len(output_size) != 2 or min(output_size) < 1:
            raise ValueError(f"output_size must be positive (height, width), got {output_size}")

        cube = F.interpolate(cube.float(), size=output_size, mode="bilinear", align_corners=False)
        features = self.spatial_blocks(self.stems[spec.name](cube))
        return [
            head(features).flatten(2).transpose(1, 2).contiguous() for head in self.prompt_heads
        ]
