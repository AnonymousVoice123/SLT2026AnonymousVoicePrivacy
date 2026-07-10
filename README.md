# Projected CFG Seed-VC V2 Anonymization

This repository provides a compact Seed-VC V2 based speech anonymization pipeline with projected classifier-free guidance (CFG). It supports switching projection modes from a single shell script, without manually editing the CFM code.

The implementation is adapted from [Seed-VC](https://github.com/Plachtaa/seed-vc).

## Files

```text
.
├── inference_v2.py
├── run_anonymize_one.sh
├── download_models.sh
├── requirements.txt
├── configs/
│   └── v2/
│       └── vc_wrapper.yaml
└── modules/
    └── v2/
        ├── cfm.py
        └── vc_wrapper.py
```

If your project keeps more Seed-VC modules, keep the original `modules/` structure required by `configs/v2/vc_wrapper.yaml`.

## 1. Create Environment

Create a clean environment:

```bash
conda create -n seedvc_proj python=3.10 -y
conda activate seedvc_proj
```

Install PyTorch according to your CUDA version.

For CUDA 12.1:

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8:

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Install `ffmpeg`:

```bash
conda install -c conda-forge ffmpeg -y
```

The provided `requirements.txt` does **not** include PyTorch. Install PyTorch separately as shown above.

## 2. Download Models

Run:

```bash
bash download_models.sh
```

If Hugging Face is not accessible:

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash download_models.sh
```

The expected checkpoint layout is:

```text
pretrained/
├── seedvc_v2/
│   └── v2/
│       ├── cfm_small.pth
│       └── ar_base.pth
├── astral/
│   ├── bsq32/
│   │   └── bsq32_light.pth
│   └── bsq2048/
│       └── bsq2048_light.pth
└── campplus/
    └── campplus_cn_common.bin
```

If automatic download fails, manually place the checkpoints in the paths above.

## 3. Run One Audio Example

Edit `run.sh`:

```bash
SOURCE_AUDIO="/path/to/source.wav"
AR_TARGET_AUDIO="/path/to/ar_target.wav"
OUTPUT_DIR="./outputs_single"
PROJECTION_PRESET="smooth21"
```

Then run:

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh
```

Or override paths directly from the command line:

```bash
CUDA_VISIBLE_DEVICES=0 \
SOURCE_AUDIO="/path/to/source.wav" \
AR_TARGET_AUDIO="/path/to/ar_target.wav" \
PROJECTION_PRESET="smooth21" \
bash run.sh
```

## 4. Projection Presets

Supported presets:

```text
global    : utterance-level projection
token     : frame/token-level projection
smooth21  : smoothed projection with kernel size 21
```

Default recommended setting:

```bash
PROJECTION_PRESET="smooth21"
```

Default preset values:

```text
global:
  scale1 = 1.2
  scale2 = 2.9

token:
  scale1 = 0.1
  scale2 = 1.3

smooth21:
  scale1 = 1.1
  scale2 = 2.8
```

You can override the scales:

```bash
CUDA_VISIBLE_DEVICES=0 \
SOURCE_AUDIO="/path/to/source.wav" \
AR_TARGET_AUDIO="/path/to/ar_target.wav" \
PROJECTION_PRESET="smooth21" \
SCALE1=1.1 \
SCALE2=2.8 \
bash run.sh
```

## 5. What the Script Does

`run.sh` runs `inference_v2.py` approximately as:

```bash
python inference_v2.py \
  --source "$SOURCE_AUDIO" \
  --target "$SOURCE_AUDIO" \
  --ar-target-audio-path "$AR_TARGET_AUDIO" \
  --output "$OUTPUT_DIR" \
  --intelligibility-cfg-rate "$SCALE1" \
  --similarity-cfg-rate "$SCALE2" \
  --projection-mode "$PROJECTION_MODE" \
  --smooth-kernel "$SMOOTH_KERNEL" \
  --convert-style true \
  --anonymization-only false
```

This means:

```text
SOURCE_AUDIO:
  audio to be anonymized

AR_TARGET_AUDIO:
  target audio used only for AR target-conditioned token conversion

--target "$SOURCE_AUDIO":
  CFM prompt/style uses the source audio itself
```

## 6. Direct Python Command

You can also run directly:

```bash
python inference_v2.py \
  --source /path/to/source.wav \
  --target /path/to/source.wav \
  --ar-target-audio-path /path/to/ar_target.wav \
  --output ./outputs_single \
  --diffusion-steps 30 \
  --length-adjust 1.0 \
  --intelligibility-cfg-rate 1.1 \
  --similarity-cfg-rate 2.8 \
  --projection-mode smooth \
  --smooth-kernel 21 \
  --top-p 0.9 \
  --temperature 1.0 \
  --repetition-penalty 1.0 \
  --convert-style true \
  --anonymization-only false
```

## 7. Troubleshooting

Check shell syntax:

```bash
bash -n run.sh
```

Check Python syntax:

```bash
python -m py_compile inference_v2.py modules/v2/cfm.py modules/v2/vc_wrapper.py
```

If checkpoint files are missing, run:

```bash
bash download_models.sh
```

If Hugging Face cannot be reached, use:

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash download_models.sh
```

## 8. License and Acknowledgement

This code is adapted from Seed-VC. Please keep the original Seed-VC license and attribution when redistributing modified code.

Recommended repository files:

```text
LICENSE
NOTICE
README.md
```

Do not upload pretrained checkpoints or generated audio files to GitHub. Add them to `.gitignore`:

```text
pretrained/
outputs/
*.wav
*.mp3
*.flac
*.pth
*.pt
*.ckpt
```
