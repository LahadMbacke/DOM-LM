#!/bin/bash
#SBATCH --job-name=domlm_pretrain
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu_p13
#SBATCH -A hcg@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread

# ── Environment ──────────────────────────────────────────────────────────────
set -e
cd $SCRATCH

module purge
module load python/3.9.12

source $SCRATCH/DOM-LM/DOM-LM-py310/bin/activate
# pip install -q -r $SCRATCH/DOM-LM/requirements.txt

export HF_HOME="$SCRATCH/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# ── Entraînement distribué (4 GPUs) ─────────────────────────────────────────
# Params papier : bs_total=24, lr=1e-4, warmup=0.5 époque, 5 époques
# Avec 4 GPUs  : per_device_bs=6, grad_accum=1 → effective_bs=24

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --master_addr=localhost \
    --master_port=29500 \
    $SCRATCH/DOM-LM/src/train.py \
        --data_dir      $SCRATCH/DOM-LM/data/swde_preprocessed \
        --output_dir    $SCRATCH/DOM-LM/results \
        --domains       auto,book,camera,job,movie,nbaplayer,restaurant,university \
        --epochs        5 \
        --batch_size    6 \
        --grad_accum    1 \
        --lr            1e-4 \
        --warmup_ratio  0.1 \
        --fp16

echo "Training done."
