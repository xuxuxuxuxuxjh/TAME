#!/bin/bash
#SBATCH --job-name=TAME-IMPUTE
#SBATCH --account=pcon0023
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nextgen
#SBATCH --mem=200G
#
# Submit:
#   cd /users/PCON0023/xuxuxuxuxu/TAME/code/TAME
#   sbatch run_impute_best_a100.sbatch
#
# Output:
#   Writes NEW imputed *.csv/*.npy into:
#     /users/PCON0023/xuxuxuxuxu/TAME/result/MIMIC/imputation_result/best_ckpt_<SLURM_JOB_ID>/
#   Does NOT overwrite train_groundtruth/train_with_missing.

set -euo pipefail

module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate postgre

# Fix common HPC runtime mismatch (GLIBCXX_* not found)
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Avoid importing torch._dynamo/sympy stack during inference on clusters where sympy/gmpy2 may be problematic
export TORCH_DISABLE_DYNAMO=1

# Avoid mixing user-site packages (~/.local) into this conda env
export PYTHONNOUSERSITE=1

cd /users/PCON0023/xuxuxuxuxu/TAME/code/TAME

echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch,sys; print('python',sys.version); print('torch',torch.__version__); print('torch.version.cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('ngpu', torch.cuda.device_count())"
nvidia-smi || true

# Fail fast if torch is CPU-only (prevents silently running on CPU for days).
python - <<'PY'
import torch, sys
print("torch.version.cuda", torch.version.cuda)
print("torch.cuda.is_available", torch.cuda.is_available())
if torch.version.cuda is None:
    raise SystemExit("ERROR: CPU-only torch build detected (torch.version.cuda is None). Re-run setup_postgre_env.sh to install pytorch-cuda.")
PY

CKPT="/users/PCON0023/xuxuxuxuxu/TAME/data/MIMIC/models/best.ckpt"
OUT_DIR="/users/PCON0023/xuxuxuxuxu/TAME/result/MIMIC/imputation_result/best_ckpt_${SLURM_JOB_ID}"

echo "Checkpoint: ${CKPT}"
echo "Impute out : ${OUT_DIR}"

python main.py \
  --dataset MIMIC \
  --data-dir /users/PCON0023/xuxuxuxuxu/TAME/data \
  --file-dir /users/PCON0023/xuxuxuxuxu/TAME/file \
  --result-dir /users/PCON0023/xuxuxuxuxu/TAME/result \
  --phase test \
  --resume "${CKPT}" \
  --impute-dir "${OUT_DIR}" \
  --model tame \
  --use-ta 1 \
  --use-ve 1 \
  --use-mm 0 \
  --split-num 4000 \
  --batch-size 64 \
  --workers 12

echo "Done. Outputs in: ${OUT_DIR}"


