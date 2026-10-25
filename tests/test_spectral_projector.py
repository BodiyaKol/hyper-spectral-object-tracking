import pytest

torch = pytest.importorskip("torch")

from hsi_tracking.models.spectral_projector import SpectralProjector


@pytest.mark.parametrize("sensor,channels", [("VIS", 16), ("NIR", 25), ("RedNIR", 15)])
def test_projector_outputs_three_channels_and_keeps_gradients(sensor, channels):
    projector = SpectralProjector()
    cube = torch.rand(2, channels, 8, 10)

    pseudo_rgb = projector(cube, sensor)

    assert pseudo_rgb.shape == (2, 3, 8, 10)
    pseudo_rgb.mean().backward()
    assert projector.projectors[sensor].weight.grad is not None


def test_projector_rejects_wrong_sensor_channel_count():
    projector = SpectralProjector()

    with pytest.raises(ValueError, match="expects 25 channels"):
        projector(torch.rand(1, 16, 8, 8), "NIR")
