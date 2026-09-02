#!/bin/bash
# =============================================================================
# Qwen3-8B  —  LID few-shot inference (local model, no OpenRouter)
# =============================================================================
#
# Identical resource profile to qwen_lid_zeroshot.sh; only the prompt differs.
# See that script for the full resource rationale.
# =============================================================================

#SBATCH --job-name=qwen_lid_few
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

export HF_HOME=/fs/nexus-scratch/nlpa/hf_cache

cd "$REPO/LID_and_NER/LID"

python qwen_lid_local.py \
    --prompt  prompts/lid_prompt_few_shot.txt \
    --dataset ../data_input/cleaned_annotated_dataset.csv

echo "Job finished: $SLURM_JOB_ID"
