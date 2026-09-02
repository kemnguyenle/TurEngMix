#!/bin/bash
# =============================================================================
# Encoder fine-tuning  —  LID  —  BERTurk (dbmdz/bert-base-turkish-cased)
# =============================================================================
#
# Resource choices
# ----------------
# gres=gpu:rtxa6000:1   BERT-base (~110 M params) fp16 training ≈ 1-2 GB VRAM;
#                       A6000 is far more than needed but is the pinned GPU.
# cpus-per-task=4       Trainer DataLoader workers (default 4) + main process.
# mem=32G               Dataset + model + optimizer states in RAM; well within
#                       the default-QOS 32 G ceiling.
# time=0-03:00:00       10 epochs × ~750 steps × ~0.05 s/step ≈ 6 min;
#                       3 h provides a large buffer for slow I/O at startup.
# qos=default           Caps: 1 GPU / 4 CPU / 32 G / 3 days  ✓ all fit.
#
# PREREQUISITE: run 01_prepare_dataset.py once before submitting training jobs.
#   cd $REPO/LID_and_NER/encoder_finetuning/encoders_LID && python 01_prepare_dataset.py
# =============================================================================

#SBATCH --job-name=lid_berturk
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
MODEL_OUT="/fs/nexus-scratch/nlpa/models/berturk_lid"

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: processed dataset not found at $DATA_DIR"
    echo "Run:  cd $SCRIPT_DIR && python 01_prepare_dataset.py"
    exit 1
fi

python "$SCRIPT_DIR/02_train_lid.py" \
    --model_name  dbmdz/bert-base-turkish-cased \
    --data_dir    "$DATA_DIR" \
    --output_dir  "$MODEL_OUT" \
    --epochs      10 \
    --batch_size  16 \
    --learning_rate 2e-5 \
    --weight_decay  0.01 \
    --seed        42 \
    --fp16

echo "Job finished: $SLURM_JOB_ID  →  $MODEL_OUT"
