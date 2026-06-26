#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-5}"
export PYTHON="${PYTHON:-/home/zlong/anaconda3/envs/clt-npu-py39/bin/python}"

if [[ "${1:-}" == "python" || "${1:-}" == "python3" ]]; then
  shift
  set -- "${PYTHON}" "$@"
fi

exec "${SCRIPT_DIR}/with_ascend_env.sh" "$@"
