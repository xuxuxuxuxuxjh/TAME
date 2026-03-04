#!/bin/bash
#SBATCH --job-name=TAME-MIMIC
#SBATCH --account=pcon0023
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nextgen
#SBATCH --mem=200G
#
# Usage:
#   sbatch run_mimic_a100.sbatch
#
# This assumes you already ran full preprocessing and have:
#   /users/PCON0023/xuxuxuxuxu/TAME/data/MIMIC/train_{groundtruth,with_missing}
#   /users/PCON0023/xuxuxuxuxu/TAME/file/MIMIC_*.json

set -euo pipefail

# NOTE:
# - Correct way: `sbatch run_mimic_a100.sbatch`
# - If you run `bash run_mimic_a100.sbatch` on login, you will NOT get a GPU allocation.

module load miniconda3/24.1.2-py310

# Robust conda activate in batch mode
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate postgre

# Prefer conda env runtime libs over module base (fix GLIBCXX_* errors).
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Avoid mixing user-site packages (~/.local) into this conda env
export PYTHONNOUSERSITE=1

cd /users/PCON0023/xuxuxuxuxu/TAME/code/TAME

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch,sys; print('python',sys.version); print('torch',torch.__version__); print('torch.version.cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('ngpu', torch.cuda.device_count())"

# Train
python main.py \
  --dataset MIMIC \
  --data-dir /users/PCON0023/xuxuxuxuxu/TAME/data \
  --file-dir /users/PCON0023/xuxuxuxuxu/TAME/file \
  --result-dir /users/PCON0023/xuxuxuxuxu/TAME/result \
  --phase train \
  --model tame \
  --use-ta 1 \
  --use-ve 1 \
  --use-mm 0 \
  --split-num 4000 \
  --batch-size 64 \
  --workers 12 \
  --epochs 30 \
  --lr 0.001


