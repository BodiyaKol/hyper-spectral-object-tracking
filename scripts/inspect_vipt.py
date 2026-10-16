"""Report ViPT source, configuration, checkpoint, and runtime details."""

import argparse
import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = PROJECT_ROOT / "third_party" / "ViPT"
PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "PyYAML": "yaml",
    "easydict": "easydict",
    "opencv-python": "cv2",
    "numpy": "numpy",
    "pandas": "pandas",
    "tqdm": "tqdm",
    "pycocotools": "pycocotools",
    "jpeg4py": "jpeg4py",
    "scipy": "scipy",
    "timm": "timm",
    "tb-nightly": "tensorboard",
    "lmdb": "lmdb",
    "visdom": "visdom",
    "vot-toolkit-python": "vot",
}


def print_section(title):
    print("\n" + "=" * 18 + " " + title + " " + "=" * 18)


def report_runtime():
    print_section("Python environment")
    print("Python:", sys.version.replace("\n", " "))
    print("Executable:", sys.executable)
    print("CONDA_DEFAULT_ENV:", os.environ.get("CONDA_DEFAULT_ENV", "not set"))

    try:
        from importlib import metadata
    except ImportError:
        try:
            import importlib_metadata as metadata
        except ImportError:
            metadata = None

    for distribution, module in PACKAGES.items():
        available = importlib.util.find_spec(module) is not None
        version = ""
        if available and metadata is not None:
            try:
                version = " " + metadata.version(distribution)
            except metadata.PackageNotFoundError:
                pass
        print(
            "{:<20} {}{}".format(
                distribution + ":", "available" if available else "missing", version
            )
        )

    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            print("torch.cuda.is_available():", torch.cuda.is_available())
            if torch.cuda.is_available():
                print("CUDA runtime:", torch.version.cuda)
                print("GPU:", torch.cuda.get_device_name(0))
        except (ImportError, OSError, RuntimeError) as error:
            print("PyTorch runtime check failed:", repr(error))


def report_git(repo):
    print_section("ViPT checkout")
    print("Path:", repo)
    if not (repo / ".git").exists():
        print("Git checkout: not found")
        return
    for args, label in (
        (["rev-parse", "--short", "HEAD"], "Commit"),
        (["branch", "--show-current"], "Branch"),
    ):
        try:
            value = subprocess.check_output(
                ["git", "-C", str(repo)] + args,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            value = "unknown"
        print(f"{label}: {value}")


def report_configs(repo):
    print_section("Available ViPT configurations")
    config_dir = repo / "experiments" / "vipt"
    config_paths = sorted(config_dir.glob("*.yaml"))
    if not config_paths:
        print("No YAML configs found in", config_dir)
        return

    try:
        import yaml
    except ImportError:
        yaml = None

    for path in config_paths:
        print(f"\n{path.stem}:")
        if yaml is None:
            print("  PyYAML is missing; cannot parse config values.")
            continue
        try:
            with path.open(encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file) or {}
        except (OSError, yaml.YAMLError) as error:
            print("  YAML parse failed:", repr(error))
            continue

        data = config.get("DATA", {})
        model = config.get("MODEL", {})
        backbone = model.get("BACKBONE", {})
        train = config.get("TRAIN", {})
        prompt = train.get("PROMPT", {})
        test = config.get("TEST", {})
        print("  train datasets:", data.get("TRAIN", {}).get("DATASETS_NAME", []))
        print("  input XTYPE:", data.get("XTYPE", "rgbrgb (evaluation fallback)"))
        print("  prompt:", prompt.get("TYPE", "not set"))
        print("  backbone:", backbone.get("TYPE", "not set"))
        print("  foundation checkpoint:", model.get("PRETRAIN_FILE", "not set"))
        print(
            "  template/search size:",
            test.get("TEMPLATE_SIZE", "?"),
            "/",
            test.get("SEARCH_SIZE", "?"),
        )
        print("  test epoch:", test.get("EPOCH", "not set"))


def report_api(repo):
    print_section("Inference API and input format")
    tracker_path = repo / "lib" / "test" / "tracker" / "vipt.py"
    if tracker_path.is_file():
        source = tracker_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(tracker_path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "ViPTTrack":
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name in {
                        "initialize",
                        "track",
                    }:
                        arguments = [argument.arg for argument in method.args.args]
                        print("{}.{}({})".format(node.name, method.name, ", ".join(arguments)))

    data_utils_path = repo / "lib" / "test" / "tracker" / "data_utils.py"
    if data_utils_path.is_file():
        lines = data_utils_path.read_text(encoding="utf-8").splitlines()
        start = next(
            (index for index, line in enumerate(lines) if line.startswith("class PreprocessorMM")),
            None,
        )
        if start is not None:
            end = next(
                (
                    index
                    for index in range(start + 1, len(lines))
                    if lines[index].startswith("class ")
                ),
                len(lines),
            )
            print("\nPreprocessorMM source (normalization and tensor layout):")
            for line in lines[start:end]:
                print("  " + line)

    depth_utils_path = repo / "lib" / "train" / "dataset" / "depth_utils.py"
    if depth_utils_path.is_file():
        lines = depth_utils_path.read_text(encoding="utf-8").splitlines()
        start = next(
            (index for index, line in enumerate(lines) if line.startswith("def get_x_frame")), None
        )
        if start is not None:
            end = next(
                (
                    index
                    for index in range(start + 1, len(lines))
                    if lines[index].startswith("def ")
                ),
                len(lines),
            )
            print("\nget_x_frame implementation:")
            for line in lines[start:end]:
                print("  " + line)


def report_checkpoints(repo):
    print_section("Checkpoint files")
    found = False
    for folder_name in ("models", "pretrained"):
        folder = repo / folder_name
        checkpoints = sorted(
            path
            for path in folder.glob("*")
            if path.is_file() and path.suffix.lower() in {".pth", ".tar", ".pt"}
        )
        print(f"{folder_name}: {len(checkpoints)}")
        for checkpoint in checkpoints:
            found = True
            print(f"  {checkpoint.name} ({checkpoint.stat().st_size / (1024 * 1024):.1f} MiB)")
    if not found:
        print("No model weights found. The README links checkpoints separately from source code.")


def report_install_script(repo):
    print_section("Install script commands")
    install_path = repo / "install_vipt.sh"
    if not install_path.is_file():
        print("install_vipt.sh not found")
        return
    for line in install_path.read_text(encoding="utf-8").splitlines():
        command = line.strip()
        if command.startswith(("conda install", "pip install", "apt-get install")):
            print(" ", command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=DEFAULT_REPO, help="Path to the local ViPT checkout"
    )
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()

    if not repo.is_dir():
        parser.error(f"ViPT directory does not exist: {repo}")

    report_runtime()
    report_git(repo)
    report_configs(repo)
    report_api(repo)
    report_checkpoints(repo)
    report_install_script(repo)

    print_section("HSI compatibility note")
    print("Available configs are RGB-D, RGB-E, and RGB-T; no HSI config is included upstream.")
    print("The ViPT preprocessor expects RGB plus a second, spatially aligned 3-channel modality.")
    print(
        "Use the report above to choose a checkpoint and plan an HSI adapter; do not treat RGB-only as HSI inference."
    )


if __name__ == "__main__":
    main()
