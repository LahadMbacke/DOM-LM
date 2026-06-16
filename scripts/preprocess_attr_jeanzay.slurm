#!/bin/bash
#SBATCH --job-name=domlm_preprocess_attr
#SBATCH --output=logs/preprocess_attr_%j.out
#SBATCH --error=logs/preprocess_attr_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=6:00:00
#SBATCH --partition=prepost
#SBATCH -A hcg@cpu

set -e
cd $SCRATCH/DOM-LM

module purge
module load python/3.9.12
source $SCRATCH/DOM-LM/DOM-LM-py310/bin/activate

export HF_HOME="$SCRATCH/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p logs

python src/preprocess_swde.py \
    --task       attr_extract \
    --input_dir  $SCRATCH/DOM-LM/data/swde_html \
    --config     $SCRATCH/DOM-LM/domlm-config/config.json \
    --output_dir $SCRATCH/DOM-LM/data/swde_attr_preprocessed \
    --domains    auto,book,camera,job,movie,nbaplayer,restaurant,university

echo "Attribute extraction preprocessing done."
