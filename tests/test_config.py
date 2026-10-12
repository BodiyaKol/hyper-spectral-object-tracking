from pathlib import Path

from hsi_tracking.config import load_config


def test_default_config_matches_proposal() -> None:
    config_path = Path(__file__).parents[1] / "config" / "default.yaml"

    config = load_config(config_path)

    assert config["data"]["sensors"] == {
        "VIS": {"channels": 16},
        "NIR": {"channels": 25},
        "RedNIR": {"channels": 15},
    }
    assert config["model"]["trainable_parameter_ratio_limit"] < 0.03
    assert config["evaluation"]["metrics"] == ["AUC", "DP@20", "FPS"]