#!/usr/bin/env bash
set -euo pipefail

ASCEND_TOOLKIT_ROOT="${ASCEND_TOOLKIT_ROOT:-/usr/local/Ascend/ascend-toolkit/8.2.RC1}"

export ASCEND_HOME_PATH="${ASCEND_TOOLKIT_ROOT}"
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_ROOT}"
export ASCEND_AICPU_PATH="${ASCEND_TOOLKIT_ROOT}"
export ASCEND_OPP_PATH="${ASCEND_TOOLKIT_ROOT}/opp"
export PYTHONPATH="${ASCEND_TOOLKIT_ROOT}/python/site-packages:${ASCEND_TOOLKIT_ROOT}/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${ASCEND_TOOLKIT_ROOT}/lib64:${ASCEND_TOOLKIT_ROOT}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_ROOT}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_ROOT}/hccl/lib64:${ASCEND_TOOLKIT_ROOT}/fwkacllib/lib64:${ASCEND_TOOLKIT_ROOT}/aarch64-linux/lib64:${ASCEND_TOOLKIT_ROOT}/runtime/lib64:${ASCEND_TOOLKIT_ROOT}/tools/aml/lib64:${ASCEND_TOOLKIT_ROOT}/tools/aml/lib64/plugin:${LD_LIBRARY_PATH:-}"

exec "$@"
