#!/bin/bash
set -euo pipefail

# Configure the `postgre` conda environment for running TAME (py310 + pytorch-cuda + deps).
#
# Usage (login node is fine for installation):
#   bash setup_postgre_env.sh
#
# Then (on GPU allocation):
#   python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"

module load miniconda3/24.1.2-py310

source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_NAME="postgre"
ENV_FILE="$(cd "$(dirname "$0")" && pwd)/env_postgre.yml"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] Updating existing conda env: ${ENV_NAME}"
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
  echo "[setup] Creating conda env: ${ENV_NAME}"
  conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi

conda activate "${ENV_NAME}"

# Prefer the conda env runtime libs over the module's base libs (fixes GLIBCXX_* errors).
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "[setup] Verifying imports..."
python - <<'PY'
import sys
import numpy as np
import tqdm
import pandas as pd
import sklearn
import torch
print("python", sys.version)
print("numpy", np.__version__)
print("pandas", pd.__version__)
print("tqdm", tqdm.__version__)
print("sklearn", sklearn.__version__)
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("torch_file", torch.__file__)
PY

echo "[setup] Ensuring CUDA-enabled PyTorch build is installed (torch.version.cuda != None)..."
CUDA_OK=0
python - <<'PY' || CUDA_OK=$?
import torch
import sys
cuda_ver = torch.version.cuda
print("torch.version.cuda =", cuda_ver)
sys.exit(0 if cuda_ver is not None else 2)
PY

if [[ "${CUDA_OK}" -ne 0 ]]; then
  echo "[setup] Detected CPU-only torch build. Reinstalling CUDA-enabled PyTorch via conda..."
  # Remove any pip-installed torch to avoid shadowing conda packages.
  python -m pip uninstall -y torch torchvision torchaudio || true
  conda install -y pytorch pytorch-cuda=12.1 torchvision torchaudio -c pytorch -c nvidia
fi

echo "[setup] Final torch check..."
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("torch_file", torch.__file__)
PY

echo "[setup] Done."

