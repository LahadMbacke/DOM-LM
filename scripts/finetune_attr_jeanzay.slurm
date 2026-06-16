#!/bin/bash
#SBATCH --job-name=domlm_finetune_attr
#SBATCH --output=logs/finetune_attr_%j.out
#SBATCH --error=logs/finetune_attr_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p13
#SBATCH -A hcg@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread

set -e
cd $SCRATCH/DOM-LM

module purge
module load python/3.9.12
source $SCRATCH/DOM-LM/DOM-LM-py310/bin/activate

export HF_HOME="$SCRATCH/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p logs

# ── Update this to your latest pre-trained checkpoint ────────────────────────
MODEL_PATH=$SCRATCH/DOM-LM/results/$(ls $SCRATCH/DOM-LM/results/ | grep checkpoint | sort -V | tail -1)
echo "Using model: $MODEL_PATH"

SETTING=${SETTING:-few_shot}   # override with: sbatch --export=SETTING=zero_shot ...

for DOMAIN in auto book camera job movie nbaplayer restaurant university; do
    echo "=== Domain: $DOMAIN  (setting: $SETTING) ==="
    python $SCRATCH/DOM-LM/src/finetune_attr.py \
        --data_dir     $SCRATCH/DOM-LM/data/swde_attr_preprocessed \
        --model_path   $MODEL_PATH \
        --output_dir   $SCRATCH/DOM-LM/results_attr \
        --domain       $DOMAIN \
        --setting      $SETTING \
        --n_train      10 \
        --epochs       5 \
        --batch_size   4 \
        --lr           5e-5 \
        --warmup_ratio 0.1
done

# Aggregate all domains
python - <<'EOF'
import json, glob
results = []
for p in glob.glob("$SCRATCH/DOM-LM/results_attr/*/avg_*.json"):
    with open(p) as f:
        results.append(json.load(f))
if results:
    avg = sum(r["avg_f1"] for r in results) / len(results)
    print(f"\n=== Overall avg F1 across {len(results)} domains: {avg:.4f} ===")
    for r in sorted(results, key=lambda x: x["domain"]):
        print(f"  {r['domain']:12s}  {r['avg_f1']:.4f}")
EOF

echo "Fine-tuning done."
