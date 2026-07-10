#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Projected Seed-VC V2: one-audio anonymization pipeline
# ============================================================
# Usage: edit SOURCE_AUDIO, AR_TARGET_AUDIO and PROJECTION_PRESET below, then run:
#   bash run.sh
# ============================================================

# ---------------- User inputs ----------------
# Source audio to be anonymized.
SOURCE_AUDIO="${SOURCE_AUDIO:-/path/to/source.wav}"

# AR target audio. This is used as AR context only.
AR_TARGET_AUDIO="${AR_TARGET_AUDIO:-/path/to/ar_target.wav}"

# Output directory.
OUTPUT_DIR="${OUTPUT_DIR:-./outputs_single}"

# Preset choices:
#   global   : utterance-level projection
#   token    : token/frame-level projection
#   smooth21 : smoothed projection with kernel size 21
PROJECTION_PRESET="${PROJECTION_PRESET:-smooth21}"

# Optional custom scales. Leave empty to use the preset default.
SCALE1="${SCALE1:-}"
SCALE2="${SCALE2:-}"

# Inference parameters.
DIFFUSION_STEPS="${DIFFUSION_STEPS:-30}"
LENGTH_ADJUST="${LENGTH_ADJUST:-1.0}"
TOP_P="${TOP_P:-0.9}"
TEMPERATURE="${TEMPERATURE:-1.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
COMPILE="${COMPILE:-false}"

# Optional custom checkpoints. Leave empty to use paths inside vc_wrapper.py.
AR_CHECKPOINT_PATH="${AR_CHECKPOINT_PATH:-}"
CFM_CHECKPOINT_PATH="${CFM_CHECKPOINT_PATH:-}"

# ---------------- Preset table ----------------
# You can edit the default SCALE1/SCALE2 here if you later update your paper's best configs.
case "$PROJECTION_PRESET" in
    global)
        PROJECTION_MODE="global"
        SMOOTH_KERNEL=21
        DEFAULT_SCALE1=1.2
        DEFAULT_SCALE2=2.9
        ;;

    token)
        PROJECTION_MODE="token"
        SMOOTH_KERNEL=21
        DEFAULT_SCALE1=0.1
        DEFAULT_SCALE2=1.3
        ;;

    smooth|smooth21)
        PROJECTION_MODE="smooth"
        SMOOTH_KERNEL=21
        DEFAULT_SCALE1=1.1
        DEFAULT_SCALE2=2.8
        ;;

    
    *)
        echo "ERROR: unknown PROJECTION_PRESET=$PROJECTION_PRESET"
        echo "Valid choices: global, token, smooth21"
        exit 1
        ;;
esac

SCALE1="${SCALE1:-$DEFAULT_SCALE1}"
SCALE2="${SCALE2:-$DEFAULT_SCALE2}"

# ---------------- Validation ----------------
if [[ "$SOURCE_AUDIO" == "/path/to/source.wav" ]]; then
    echo "ERROR: please set SOURCE_AUDIO in this file or as an environment variable."
    exit 1
fi

if [[ "$AR_TARGET_AUDIO" == "/path/to/ar_target.wav" ]]; then
    echo "ERROR: please set AR_TARGET_AUDIO in this file or as an environment variable."
    exit 1
fi

if [[ ! -f "$SOURCE_AUDIO" ]]; then
    echo "ERROR: source audio not found: $SOURCE_AUDIO"
    exit 1
fi

if [[ ! -f "$AR_TARGET_AUDIO" ]]; then
    echo "ERROR: AR target audio not found: $AR_TARGET_AUDIO"
    exit 1
fi

if [[ ! -f "inference_v2.py" ]]; then
    echo "ERROR: please run this script from the Seed-VC repo root, where inference_v2.py exists."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# In this paper pipeline, CFM prompt/style use self-source audio; AR uses AR_TARGET_AUDIO.
CFM_TARGET_AUDIO="$SOURCE_AUDIO"

echo "============================================================"
echo "Projected Seed-VC V2 one-audio anonymization"
echo "============================================================"
echo "SOURCE_AUDIO        : $SOURCE_AUDIO"
echo "CFM_TARGET_AUDIO    : $CFM_TARGET_AUDIO"
echo "AR_TARGET_AUDIO     : $AR_TARGET_AUDIO"
echo "OUTPUT_DIR          : $OUTPUT_DIR"
echo "PROJECTION_PRESET   : $PROJECTION_PRESET"
echo "PROJECTION_MODE     : $PROJECTION_MODE"
echo "SMOOTH_KERNEL       : $SMOOTH_KERNEL"
echo "SCALE1              : $SCALE1"
echo "SCALE2              : $SCALE2"
echo "DIFFUSION_STEPS     : $DIFFUSION_STEPS"
echo "LENGTH_ADJUST       : $LENGTH_ADJUST"
echo "TOP_P               : $TOP_P"
echo "TEMPERATURE         : $TEMPERATURE"
echo "REPETITION_PENALTY  : $REPETITION_PENALTY"
echo "COMPILE             : $COMPILE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "============================================================"

CMD=(
    python3 inference_v2.py
    --source "$SOURCE_AUDIO"
    --target "$CFM_TARGET_AUDIO"
    --ar-target-audio-path "$AR_TARGET_AUDIO"
    --output "$OUTPUT_DIR"
    --diffusion-steps "$DIFFUSION_STEPS"
    --length-adjust "$LENGTH_ADJUST"
    --intelligibility-cfg-rate "$SCALE1"
    --similarity-cfg-rate "$SCALE2"
    --projection-mode "$PROJECTION_MODE"
    --smooth-kernel "$SMOOTH_KERNEL"
    --top-p "$TOP_P"
    --temperature "$TEMPERATURE"
    --repetition-penalty "$REPETITION_PENALTY"
    --convert-style true
    --anonymization-only false
    --compile "$COMPILE"
)

if [[ -n "$AR_CHECKPOINT_PATH" ]]; then
    CMD+=(--ar-checkpoint-path "$AR_CHECKPOINT_PATH")
fi

if [[ -n "$CFM_CHECKPOINT_PATH" ]]; then
    CMD+=(--cfm-checkpoint-path "$CFM_CHECKPOINT_PATH")
fi

echo
echo "Running command:"
printf ' %q' "${CMD[@]}"
echo
echo

"${CMD[@]}"

echo
echo "Done. Output directory: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
