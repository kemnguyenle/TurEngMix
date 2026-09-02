#!/bin/bash
# =============================================================================
# Encoder fine-tuning  —  NER  —  BERTurk (dbmdz/bert-base-turkish-cased)
# =============================================================================
#
# NER uses span-level (seqeval) F1 for checkpoint selection instead of flat
# per-token F1; the training script (02_train_ner.py) handles this.
# Resource profile identical to the LID encoder jobs; see encoder_lid_berturk.sh
# for the full rationale.
#
# PREREQUISITE: run NER 01_prepare_dataset.py once before submitting NER jobs:
#   cd $REPO/LID_and_NER/encoder_finetuning/encoders_NER && python 01_prepare_dataset.py
# =============================================================================

#SBATCH --job-name=ner_berturk
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

SCRIPT_DIR="$REPO/LID_and_NER/encoder_finetuning/encoders_NER"
DATA_DIR="$SCRIPT_DIR/processed_data"
MODEL_OUT="/fs/nexus-scratch/nlpa/models/berturk_ner"

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: processed NER dataset not found at $DATA_DIR"
    echo "Run:  cd $SCRIPT_DIR && python 01_prepare_dataset.py"
    exit 1
fi

python "$SCRIPT_DIR/02_train_ner.py" \
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
