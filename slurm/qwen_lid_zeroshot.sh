#!/bin/bash
# =============================================================================
# Qwen3-8B  —  LID zero-shot inference (local model, no OpenRouter)
# =============================================================================
#
# Resource choices
# ----------------
# gres=gpu:rtxa6000:1   Qwen3-8B in bfloat16 ≈ 16 GB VRAM; A6000 has 48 GB,
#                       leaving ample headroom for activations and KV cache.
# cpus-per-task=4       1 main + 2-3 DataLoader workers; keeps I/O off the GPU.
# mem=32G               Model weights reside in CPU memory briefly during load;
#                       32 GB is the default-QOS ceiling and is sufficient.
# time=0-02:00:00       249 docs × ~3 s/doc ≈ 12 min inference + ~2 min load;
#                       2 h gives a 10× safety margin well within the 3-day cap.
# qos=default           Caps: 1 GPU / 4 CPU / 32 G / 3 days  ✓ all fit.
# =============================================================================

#SBATCH --job-name=qwen_lid_zero
#SBATCH --account=clip
#SBATCH --partition=clip
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=/fs/nexus-scratch/nlpa/logs/%x_%j.out
#SBATCH --error=/fs/nexus-scratch/nlpa/logs/%x_%j.err

# ── User-configurable ─────────────────────────────────────────────────────────
REPO=/fs/nexus-scratch/nlpa/Who-Writes-Oyle-CMSC499
# ─────────────────────────────────────────────────────────────────────────────

set -e
mkdir -p /fs/nexus-scratch/nlpa/logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /fs/nexus-scratch/nlpa/envs/oyle

# Point HF model cache at scratch to avoid filling your home-dir quota
export HF_HOME=/fs/nexus-scratch/nlpa/hf_cache

cd "$REPO/LID_and_NER/LID"

python qwen_lid_local.py \
    --prompt  prompts/lid_prompt.txt \
    --dataset ../data_input/cleaned_annotated_dataset.csv

echo "Job finished: $SLURM_JOB_ID"
