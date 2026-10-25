"""Prepare a local ViPT checkout for CPU inference with external HSI prompts."""

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = PROJECT_ROOT / "third_party" / "ViPT"
CPU_PATCHER = PROJECT_ROOT / "scripts" / "vipt_init-false-color" / "patch_vipt_cpu.py"


def replace_once(text, old, new, label):
    if new in text:
        return text
    if text.count(old) != 1:
        raise ValueError(f"Expected one upstream source block for {label}")
    return text.replace(old, new, 1)


def write_if_changed(path, original, updated):
    if updated != original:
        path.write_text(updated, encoding="utf-8")
        print(f"Patched {path.name}")
    else:
        print(f"Already patched: {path.name}")


def patch_viptrack(path):
    original = path.read_text(encoding="utf-8")
    updated = replace_once(
        original,
        "                return_last_attn=False,\n                ):\n",
        "                return_last_attn=False,\n                external_prompts=None,\n                ):\n",
        "ViPTrack.forward signature",
    )
    updated = replace_once(
        updated,
        "                                    return_last_attn=return_last_attn, )",
        "                                    return_last_attn=return_last_attn,\n"
        "                                    external_prompts=external_prompts)",
        "ViPTrack backbone call",
    )
    write_if_changed(path, original, updated)


def patch_transformer(path):
    original = path.read_text(encoding="utf-8")
    updated = replace_once(
        original,
        "    def forward_features(self, z, x, mask_z=None, mask_x=None,\n"
        "                         ce_template_mask=None, ce_keep_rate=None,\n"
        "                         return_last_attn=False):",
        "    def forward_features(self, z, x, mask_z=None, mask_x=None,\n"
        "                         ce_template_mask=None, ce_keep_rate=None,\n"
        "                         return_last_attn=False, external_prompts=None):",
        "VisionTransformerCE.forward_features signature",
    )

    start_marker = "        B, H, W = x.shape[0], x.shape[2], x.shape[3]\n"
    end_marker = "        # attention mask handling\n"
    start = updated.find(start_marker)
    end = updated.find(end_marker, start + len(start_marker))
    if start < 0 or end < 0:
        raise ValueError("Could not locate ViPT input modality conversion")
    input_block = updated[start:end]

    if "if external_prompts is not None:" not in input_block:
        legacy_body = input_block[len(start_marker) :]
        hsi_branch = (
            start_marker
            + "        if external_prompts is not None:\n"
            + "            if z.shape[1] != 3 or x.shape[1] != 3:\n"
            + '                raise ValueError("External-prompt mode expects 3-channel projected inputs")\n'
            + "            if len(external_prompts) != len(self.blocks):\n"
            + '                raise ValueError("Expected one external prompt per transformer block")\n'
            + "            if self.ce_loc:\n"
            + '                raise ValueError("External prompts require candidate elimination disabled (CE_LOC=[])")\n'
            + "            z = self.patch_embed(z)\n"
            + "            x = self.patch_embed(x)\n"
            + "        else:\n"
            + textwrap.indent(legacy_body, "    ")
        )
        updated = updated[:start] + hsi_branch + updated[end:]
    elif "if self.ce_loc:" not in input_block:
        old = (
            "        if external_prompts is not None:\n"
            "            if z.shape[1] != 3 or x.shape[1] != 3:\n"
        )
        new = (
            "        if external_prompts is not None:\n"
            "            if self.ce_loc:\n"
            '                raise ValueError("External prompts require candidate elimination disabled (CE_LOC=[])")\n'
            "            if z.shape[1] != 3 or x.shape[1] != 3:\n"
        )
        updated = replace_once(updated, old, new, "candidate-elimination guard")

    injection_marker = "            if external_prompts is not None:\n                prompt = external_prompts[i]\n"
    if injection_marker not in updated:
        old_loop = "        for i, blk in enumerate(self.blocks):\n            '''\n            add parameters prompt from 1th layer\n"
        new_loop = (
            "        for i, blk in enumerate(self.blocks):\n"
            "            if external_prompts is not None:\n"
            "                prompt = external_prompts[i]\n"
            "                if prompt.shape != x.shape:\n"
            '                    raise ValueError("External prompt shape does not match ViPT tokens")\n'
            "                x = x + prompt.to(device=x.device, dtype=x.dtype)\n"
            "            '''\n"
            "            add parameters prompt from 1th layer\n"
        )
        updated = replace_once(updated, old_loop, new_loop, "per-layer prompt insertion")

    updated = replace_once(
        updated,
        "            if i >= 1:\n                if self.prompt_type in ['vipt_deep']:",
        "            if external_prompts is None and i >= 1:\n"
        "                if self.prompt_type in ['vipt_deep']:",
        "legacy prompts bypass in HSI mode",
    )
    old_forward = (
        "    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,\n"
        "                tnc_keep_rate=None,\n"
        "                return_last_attn=False):\n\n"
        "        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,)"
    )
    new_forward = (
        "    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,\n"
        "                tnc_keep_rate=None, return_last_attn=False, external_prompts=None):\n\n"
        "        x, aux_dict = self.forward_features(\n"
        "            z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,\n"
        "            return_last_attn=return_last_attn, external_prompts=external_prompts\n"
        "        )"
    )
    updated = replace_once(
        updated, old_forward, new_forward, "ViT.forward external-prompts argument"
    )
    write_if_changed(path, original, updated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    repo = parser.parse_args().repo.expanduser().resolve()
    if not (repo / ".git").exists():
        parser.error(f"Expected a ViPT checkout at {repo}")

    subprocess.run([sys.executable, str(CPU_PATCHER), "--repo", str(repo)], check=True)
    patch_viptrack(repo / "lib" / "models" / "vipt" / "ostrack_prompt.py")
    patch_transformer(repo / "lib" / "models" / "vipt" / "vit_ce_prompt.py")


if __name__ == "__main__":
    main()
