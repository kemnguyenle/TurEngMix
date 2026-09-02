#!/bin/bash
# =============================================================================
# Encoder fine-tuning  —  LID  —  XLM-R (FacebookAI/xlm-roberta-base)
# =============================================================================
#
# XLM-R base is ~270 M params, so slightly heavier than BERTurk but still
# trivially within A6000 capacity.  All other resource figures are identical
# to encoder_lid_berturk.sh; see that script for the full rationale.
#
# PREREQUISITE: run 01_prepare_dataset.py once (see encoder_lid_berturk.sh).
# =============================================================================

#SBATCH --job-name=lid_xlmr
#SBATCH --account=clip
#SBATCH --partition=clip
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
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

SCRIPT_DIR="$REPO/LID_and_NER/encoder_finetuning/encoders_LID"
DATA_DIR="$SCRIPT_DIR/processed_data"
MODEL_OUT="/fs/nexus-scratch/nlpa/models/xlmr_lid"

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: processed dataset not found at $DATA_DIR"
    echo "Run:  cd $SCRIPT_DIR && python 01_prepare_dataset.py"
    exit 1
fi

python "$SCRIPT_DIR/02_train_lid.py" \
    --model_name  FacebookAI/xlm-roberta-base \
    --data_dir    "$DATA_DIR" \
    --output_dir  "$MODEL_OUT" \
    --epochs      10 \
    --batch_size  16 \
    --learning_rate 2e-5 \
    --weight_decay  0.01 \
    --seed        42 \
    --fp16

echo "Job finished: $SLURM_JOB_ID  →  $MODEL_OUT"
