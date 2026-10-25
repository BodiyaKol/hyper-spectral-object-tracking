"""Learnable per-sensor 1x1 projection from HSI bands to pseudo-RGB."""

import torch
from torch import Tensor, nn

from hsi_tracking.data.x2cube import SENSORS, get_sensor_spec


class SpectralProjector(nn.Module):
    """Project HSI ``B x C x H x W`` tensors to three pseudo-color channels.

    Each sensor gets its own learnable 1x1 convolution because sensors have
    different band counts. The initial weights select low/middle/high bands;
    training can learn mixtures of all bands afterwards.
    """

    def __init__(self) -> None:
        super().__init__()
        self.projectors = nn.ModuleDict(
            {
                name: nn.Conv2d(spec.expected_bands, 3, kernel_size=1, bias=False)
                for name, spec in SENSORS.items()
            }
        )
        self.projection_bands = {}

        for name, spec in SENSORS.items():
            count = spec.expected_bands
            selected = (0, count // 2, count - 1)
            projector = self.projectors[name]
            nn.init.zeros_(projector.weight)
            with torch.no_grad():
                for output_channel, input_band in enumerate(selected):
                    projector.weight[output_channel, input_band, 0, 0] = 1.0
            self.projection_bands[name] = selected

    def forward(self, cube: Tensor, sensor: str) -> Tensor:
        """Project one sensor cube; input is BCHW and output is B3HW.

        Values remain in the sensor's raw intensity scale. Normalize them
        explicitly for display or training rather than silently rescaling here.
        """
        spec = get_sensor_spec(sensor)
        if cube.ndim != 4:
            raise ValueError(f"Expected HSI tensor in BCHW format, got shape {tuple(cube.shape)}")
        if cube.shape[1] != spec.expected_bands:
            raise ValueError(
                f"{spec.name} projector expects {spec.expected_bands} channels, got {cube.shape[1]}"
            )
        return self.projectors[spec.name](cube.float())
