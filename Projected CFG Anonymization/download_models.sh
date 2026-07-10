#!/usr/bin/env bash
set -euo pipefail

mkdir -p pretrained/seedvc_v2/v2
mkdir -p pretrained/astral/bsq32
mkdir -p pretrained/astral/bsq2048
mkdir -p pretrained/campplus

python3 - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download
import shutil

items = [
    (
        "Plachta/Seed-VC",
        "v2/cfm_small.pth",
        "pretrained/seedvc_v2/v2/cfm_small.pth",
    ),
    (
        "Plachta/Seed-VC",
        "v2/ar_base.pth",
        "pretrained/seedvc_v2/v2/ar_base.pth",
    ),
    (
        "Plachta/ASTRAL-quantization",
        "bsq32/bsq32_light.pth",
        "pretrained/astral/bsq32/bsq32_light.pth",
    ),
    (
        "Plachta/ASTRAL-quantization",
        "bsq2048/bsq2048_light.pth",
        "pretrained/astral/bsq2048/bsq2048_light.pth",
    ),
    (
        "funasr/campplus",
        "campplus_cn_common.bin",
        "pretrained/campplus/campplus_cn_common.bin",
    ),
]

for repo_id, filename, dst in items:
    dst = Path(dst)
    if dst.is_file():
        print(f"EXISTS: {dst}")
        continue

    print(f"Downloading {repo_id}/{filename}")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded, dst)
    print(f"SAVED: {dst}")

print("All required SeedVC checkpoints are ready.")
PY

find pretrained -type f | sort
