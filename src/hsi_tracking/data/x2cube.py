from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SensorSpec:
	"""Mosaic layout of one camera."""

	name: str
	block_size: int
	expected_bands: int

	@property
	def raw_bands(self) -> int:
		return self.block_size * self.block_size

	@property
	def dropped_bands(self) -> int:
		"""How many trailing raw bands are removed (1 for RedNIR, 0 otherwise)."""
		return self.raw_bands - self.expected_bands


SENSORS: dict[str, SensorSpec] = {
	"VIS": SensorSpec("VIS", block_size=4, expected_bands=16),
	"NIR": SensorSpec("NIR", block_size=5, expected_bands=25),
	"RedNIR": SensorSpec("RedNIR", block_size=4, expected_bands=15),
}


class ChannelCountError(ValueError):
	"""Raised when a cube does not have the number of bands its sensor expects."""


def get_sensor_spec(sensor: str) -> SensorSpec:
	"""Find a sensor by name, case-insensitive ("nir", "NIR" and "Nir" all work)."""
	for name, spec in SENSORS.items():
		if name.casefold() == sensor.casefold():
			return spec
	raise ValueError(f"Unknown sensor {sensor!r}, expected one of {sorted(SENSORS)}")


def load_mosaic(path: str | Path) -> np.ndarray:
	"""Read one raw mosaic frame as a 2D array, keeping its original dtype."""
	path = Path(path)
	try:
		with Image.open(path) as image:
			mode = image.mode
			mosaic = np.asarray(image)
	except OSError as exc:
		raise ValueError(f"Cannot read mosaic frame {path}: {exc}") from exc

	if mosaic.ndim == 3 and mosaic.shape[2] == 1:
		mosaic = mosaic[..., 0]
	if mosaic.ndim != 2:
		raise ValueError(
			f"{path} is not a single-channel mosaic: shape {mosaic.shape}, mode {mode}. "
			"False-color frames must be read separately, not passed to X2Cube."
		)
	return mosaic


def x2cube(mosaic: np.ndarray, block_size: int) -> np.ndarray:
	"""Convert a 2D mosaic into a (H // B, W // B, B * B) cube.

	Gives the same result as the official ``X2Cube(img, B=[B, B], skip=[B, B])``,
	written with reshape/transpose instead of index arithmetic.
	"""
	if mosaic.ndim != 2:
		raise ValueError(f"X2Cube expects a 2D mosaic, got shape {mosaic.shape}")
	if block_size < 1:
		raise ValueError(f"block_size must be positive, got {block_size}")

	height, width = mosaic.shape
	rows, cols = height // block_size, width // block_size
	if rows == 0 or cols == 0:
		raise ValueError(f"Mosaic {mosaic.shape} is smaller than one {block_size}x{block_size} block")

	cropped = mosaic[: rows * block_size, : cols * block_size]
	blocks = cropped.reshape(rows, block_size, cols, block_size)
	return blocks.transpose(0, 2, 1, 3).reshape(rows, cols, block_size * block_size)


def validate_channels(cube: np.ndarray, sensor: str) -> None:
	"""Raise ChannelCountError if the cube is not H x W x expected_bands."""
	spec = get_sensor_spec(sensor)
	if cube.ndim != 3 or cube.shape[2] != spec.expected_bands:
		raise ChannelCountError(
			f"{spec.name} cube must have {spec.expected_bands} bands, got shape {cube.shape}"
		)


def select_bands(raw_cube: np.ndarray, sensor: str) -> np.ndarray:
	"""Keep the sensor's real bands (removes the empty last RedNIR band)."""
	spec = get_sensor_spec(sensor)
	if raw_cube.ndim != 3 or raw_cube.shape[2] != spec.raw_bands:
		raise ChannelCountError(
			f"{spec.name} raw cube must have {spec.raw_bands} bands before band selection, "
			f"got shape {raw_cube.shape}"
		)
	cube = raw_cube[..., : spec.expected_bands]
	validate_channels(cube, spec.name)
	return cube


def mosaic_to_cube(mosaic: np.ndarray, sensor: str) -> np.ndarray:
	"""Full conversion: raw mosaic -> H x W x C cube with the sensor's band count."""
	spec = get_sensor_spec(sensor)
	return select_bands(x2cube(mosaic, spec.block_size), spec.name)
