"""Apply the project's CPU-inference compatibility patch to a ViPT checkout."""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = PROJECT_ROOT / "third_party" / "ViPT"


def replace_once(text, old, new, path):
    if old not in text:
        if new and new in text:
            return text
        raise ValueError(f"Expected ViPT source snippet not found in {path}")
    if text.count(old) != 1:
        raise ValueError(f"Expected one matching source snippet in {path}")
    return text.replace(old, new, 1)


def replace_many(text, old, new, count, path):
    matches = text.count(old)
    if matches == 0:
        if text.count(new) >= count:
            return text
        raise ValueError(f"Expected ViPT source snippets not found in {path}")
    if matches != count:
        raise ValueError(f"Expected {count} matching source snippets in {path}")
    return text.replace(old, new)


def patch_file(repo, relative_path, replacements, removals=()):
    path = repo / relative_path
    original = path.read_text(encoding="utf-8")
    updated = original
    for snippet in removals:
        updated = updated.replace(snippet, "", 1)
    for replacement in replacements:
        if len(replacement) == 3:
            old, new, count = replacement
            updated = replace_many(updated, old, new, count, relative_path)
        else:
            old, new = replacement
            updated = replace_once(updated, old, new, relative_path)
    if updated != original:
        path.write_text(updated, encoding="utf-8")
        print(f"Patched {relative_path}")
    else:
        print(f"Already patched: {relative_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=DEFAULT_REPO, help="Path to the local ViPT checkout"
    )
    repo = parser.parse_args().repo.expanduser().resolve()
    if not (repo / ".git").exists():
        parser.error(f"Expected a ViPT git checkout at {repo}")

    patch_file(
        repo,
        Path("lib/test/tracker/vipt.py"),
        [
            (
                "        super(ViPTTrack, self).__init__(params)\n        network = build_viptrack(params.cfg, training=False)\n",
                (
                    "        super(ViPTTrack, self).__init__(params)\n"
                    '        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
                    "        network = build_viptrack(params.cfg, training=False)\n"
                ),
            ),
            (
                "torch.load(self.params.checkpoint, map_location='cpu')['net']",
                "torch.load(self.params.checkpoint, map_location=self.device)['net']",
            ),
            ("self.network = network.cuda()", "self.network = network.to(self.device)"),
            (
                "self.preprocessor = PreprocessorMM()",
                "self.preprocessor = PreprocessorMM(self.device)",
            ),
            (
                "self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()",
                (
                    "self.output_window = hann2d(\n"
                    "            torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True\n"
                    "        ).to(self.device)"
                ),
            ),
        ],
        removals=(
            "import math\n",
            "from lib.test.tracker.vis_utils import gen_visualization\n",
            "import os\n",
            "import vot\n",
        ),
    )

    patch_file(
        repo,
        Path("lib/test/tracker/data_utils.py"),
        [
            (
                (
                    "    def __init__(self):\n"
                    "        self.mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406]).view((1, 6, 1, 1)).cuda()\n"
                    "        self.std = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225]).view((1, 6, 1, 1)).cuda()\n\n"
                    "    def process(self, img_arr: np.ndarray):\n"
                    "        # Deal with the image patch\n"
                    "        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)\n"
                ),
                (
                    "    def __init__(self, device=None):\n"
                    '        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
                    "        self.mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406]).view((1, 6, 1, 1)).to(self.device)\n"
                    "        self.std = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225]).view((1, 6, 1, 1)).to(self.device)\n\n"
                    "    def process(self, img_arr: np.ndarray):\n"
                    "        # Deal with the image patch\n"
                    "        img_tensor = torch.tensor(img_arr).to(self.device).float().permute((2, 0, 1)).unsqueeze(dim=0)\n"
                ),
            )
        ],
    )

    patch_file(
        repo,
        Path("lib/models/layers/head.py"),
        [
            (
                ".view((self.feat_sz * self.feat_sz,)).float().cuda()",
                ".view((self.feat_sz * self.feat_sz,)).float()",
                2,
            ),
            (
                "self.coord_x * prob_vec",
                "self.coord_x.to(score_map.device) * prob_vec",
            ),
            (
                "self.coord_y * prob_vec",
                "self.coord_y.to(score_map.device) * prob_vec",
            ),
        ],
    )

    patch_file(
        repo,
        Path("lib/test/tracker/basetracker.py"),
        [
            (
                "            try:\n                self.visdom = Visdom(",
                "            try:\n                from lib.vis.visdom_cus import Visdom\n\n                self.visdom = Visdom(",
            )
        ],
        removals=("\nfrom lib.vis.visdom_cus import Visdom\n",),
    )

    patch_file(
        repo,
        Path("lib/train/data/loader.py"),
        [("from torch._six import string_classes\n", "string_classes = str\n")],
    )

    patch_file(
        repo,
        Path("lib/train/data/image_loader.py"),
        [
            (
                "import jpeg4py\nimport cv2 as cv\n",
                "try:\n    import jpeg4py\nexcept (ImportError, OSError):\n    jpeg4py = None\nimport cv2 as cv\n",
            ),
            (
                'def jpeg4py_loader(path):\n    """ Image reading using jpeg4py https://github.com/ajkxyz/jpeg4py"""\n    try:',
                'def jpeg4py_loader(path):\n    """ Image reading using jpeg4py https://github.com/ajkxyz/jpeg4py"""\n    if jpeg4py is None:\n        return None\n    try:',
            ),
            (
                'def jpeg4py_loader_w_failsafe(path):\n    """ Image reading using jpeg4py https://github.com/ajkxyz/jpeg4py"""\n    try:\n        return jpeg4py.JPEG(path).decode()',
                'def jpeg4py_loader_w_failsafe(path):\n    """ Image reading using jpeg4py https://github.com/ajkxyz/jpeg4py"""\n    try:\n        if jpeg4py is None:\n            raise ImportError("jpeg4py is unavailable")\n        return jpeg4py.JPEG(path).decode()',
            ),
        ],
    )


if __name__ == "__main__":
    main()
