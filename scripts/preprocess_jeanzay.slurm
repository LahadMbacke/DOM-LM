#!/bin/bash
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p13
#SBATCH -A hcg@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread

# ── Environment ──────────────────────────────────────────────────────────────
cd $SCRATCH

module purge
module load python/3.9.12

export VENV=$SCRATCH/DOM-LM/DOM-LM-py310
export PYTHON=$VENV/bin/python
export HF_HOME="$SCRATCH/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# ── Preprocessing ─────────────────────────────────────────────────────────────
$PYTHON DOM-LM/src/preprocess_swde.py \
    --task       domlm \
    --input_dir  $SCRATCH/DOM-LM/data/swde_html \
    --config     $SCRATCH/DOM-LM/domlm-config/config.json \
    --output_dir $SCRATCH/DOM-LM/data/swde_preprocessed \
    --domains    auto,book,camera,job,movie,nbaplayer,restaurant,university \
    --num_workers $SLURM_CPUS_PER_TASK
