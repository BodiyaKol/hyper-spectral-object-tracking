import pytest

torch = pytest.importorskip("torch")

from hsi_tracking.models.prompt_adapter import CNNPromptAdapter


@pytest.mark.parametrize("sensor,channels", [("VIS", 16), ("NIR", 25), ("RedNIR", 15)])
def test_prompt_adapter_outputs_one_token_map_per_layer(sensor, channels):
    adapter = CNNPromptAdapter(embed_dim=32, num_layers=4, hidden_channels=16, norm_groups=4)
    cube = torch.rand(2, channels, 24, 32)

    prompts = adapter(cube, sensor, output_size=(3, 4))

    assert len(prompts) == 4
    assert all(prompt.shape == (2, 12, 32) for prompt in prompts)
    sum(prompt.mean() for prompt in prompts).backward()
    assert adapter.stems[sensor][0].weight.grad is not None


def test_prompt_adapter_rejects_wrong_sensor_channel_count():
    adapter = CNNPromptAdapter(embed_dim=32, num_layers=2, hidden_channels=16, norm_groups=4)

    with pytest.raises(ValueError, match="expects 25 channels"):
        adapter(torch.rand(1, 16, 8, 8), "NIR", output_size=(2, 2))
