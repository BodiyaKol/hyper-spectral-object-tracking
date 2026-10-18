
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="$ROOT/.venv-samurai"
UPSTREAM="$ROOT/third_party/SAMURAI"

cd "$ROOT"

echo "Setting up isolated SAMURAI environment..."

if [ ! -d "$ENV_DIR" ]; then
    python3 -m venv "$ENV_DIR"
fi

PYTHON="$ENV_DIR/bin/python"

"$PYTHON" -m pip install --upgrade pip setuptools wheel

if [ ! -f "$UPSTREAM/sam2/setup.py" ]; then
    mkdir -p "$ROOT/third_party"
    git clone https://github.com/yangchris11/samurai.git "$UPSTREAM"
fi

# Disable CUDA extension compilation on non-NVIDIA systems.
export SAM2_BUILD_CUDA=0

"$PYTHON" -m pip install torch torchvision
"$PYTHON" -m pip install -e "$UPSTREAM/sam2" --no-build-isolation
"$PYTHON" -m pip install matplotlib opencv-python pandas scipy loguru

echo ""
echo "SAMURAI installation complete."
echo "Activate with: source .venv-samurai/bin/activate"
echo ""
echo "Download the SAM 2.1 Hiera Tiny checkpoint from:"
echo "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
echo ""
echo "Save it to:"
echo "$UPSTREAM/sam2/checkpoints/sam2.1_hiera_tiny.pt"
