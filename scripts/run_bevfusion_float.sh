#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

conda_env="${HEAL_CONDA_ENV:-heal}"
if [[ "${1:-}" == --conda-env ]]; then
    if [[ $# -lt 2 ]]; then
        echo 'Usage: run_bevfusion_float.sh [--conda-env NAME] [demo arguments]' >&2
        exit 2
    fi
    conda_env="$2"
    shift 2
fi

conda_exe="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$conda_exe" ]]; then
    for dir in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"; do
        if [[ -x "$dir/bin/conda" ]]; then
            conda_exe="$dir/bin/conda"
            break
        fi
    done
fi
if [[ -z "$conda_exe" ]]; then
    echo 'Conda not found; set CONDA_EXE to the WSL Conda executable.' >&2
    exit 1
fi

exec "$conda_exe" run --no-capture-output -n "$conda_env" python -u "$root/tools/bevfusion_float_demo.py" "$@"
