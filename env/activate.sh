# Source before running project code (adapted for the release from the project's env/activate.sh).
# Keeps caches and config inside the repository and puts the project's Python/R environments on PATH.
# COLOC_BENCH_ROOT = repository root (default: the parent of this file's directory).
# Expected layout (not shipped; see docs/environment/README.md): .venv/ (Python), env/renv/ (conda R env with coloc
# 5.2.3, susieR 0.14.2, ...), env/rlib_susie016/ (susieR 0.16.6), env/rlib_mvsusie/, env/tools/ (plink2, bcftools).
P="${COLOC_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export COLOC_BENCH_ROOT="$P"
export UV_CACHE_DIR=$P/env/cache/uv PIP_CACHE_DIR=$P/env/cache/pip XDG_CACHE_HOME=$P/env/cache/xdg
export TORCH_HOME=$P/env/cache/torch MPLCONFIGDIR=$P/env/cache/mpl NUMBA_CACHE_DIR=$P/env/cache/numba
export MAMBA_ROOT_PREFIX=$P/env/mamba R_LIBS_USER=$P/env/renv/lib/R/library R_USER_CACHE_DIR=$P/env/cache/R
export TRITON_CACHE_DIR=$P/env/cache/triton CUDA_CACHE_PATH=$P/env/cache/cuda
# The original project also redirected HOME to $P/env/home (no writes to the user's home); uncomment to do the same:
# export HOME=$P/env/home R_ENVIRON_USER=$P/env/home/.Renviron R_PROFILE_USER=$P/env/home/.Rprofile
export PATH=$P/.venv/bin:$P/env/tools/bin:$P/env/renv/bin:$PATH
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NVIDIA_TF32_OVERRIDE=0
export TMPDIR="${TMPDIR:-/tmp}/coloc-bench-$(id -u)"; mkdir -p "$TMPDIR"
export PYTEST_ADDOPTS="--basetemp=$TMPDIR/pytest-$$"
export COLOC_ENV_ACTIVE=1
