#!/bin/bash
# =============================================================================
# Qwen3-8B  —  NER zero-shot inference (local model, no OpenRouter)
# =============================================================================
#
# NER output sequences are slightly longer than LID (BIO labels vs. flat
# labels) but per-doc token counts are the same, so resource needs are
# identical to the LID jobs.  See qwen_lid_zeroshot.sh for rationale.
# =============================================================================

#SBATCH --job-name=qwen_ner_zero
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

cd "$REPO/LID_and_NER/NER"

python qwen_ner_local.py \
    --prompt  prompts/ner_prompt.txt \
    --dataset ../data_input/cleaned_annotated_dataset.csv

echo "Job finished: $SLURM_JOB_ID"
