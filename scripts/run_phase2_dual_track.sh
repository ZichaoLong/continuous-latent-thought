#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEVICE="${DEVICE:-npu:0}"

PHASE2B_DATA_DIR="${PHASE2B_DATA_DIR:-data/phase2b_pointer_npu}"
PHASE2B_OUTPUT_DIR="${PHASE2B_OUTPUT_DIR:-outputs/phase2b_pointer_matrix_npu}"
PHASE2B_SEEDS="${PHASE2B_SEEDS:-0 1 2}"
PHASE2B_STEPS="${PHASE2B_STEPS:-1000}"
PHASE2B_EVAL_EXAMPLES="${PHASE2B_EVAL_EXAMPLES:-100}"

PHASE2A_DATA_DIR="${PHASE2A_DATA_DIR:-data/phase2a_ladder_scale_npu}"
PHASE2A_OUTPUT_DIR="${PHASE2A_OUTPUT_DIR:-outputs/phase2a_ladder_scale_npu}"
PHASE2A_SEEDS="${PHASE2A_SEEDS:-0 1 2}"
PHASE2A_STEPS="${PHASE2A_STEPS:-3000}"
PHASE2A_EVAL_EXAMPLES="${PHASE2A_EVAL_EXAMPLES:-200}"

cd "${ROOT_DIR}"

printf '[%s] phase2b pointer matrix start\n' "$(date -Is)"
DEVICE="${DEVICE}" \
DATA_DIR="${PHASE2B_DATA_DIR}" \
OUTPUT_DIR="${PHASE2B_OUTPUT_DIR}" \
SEEDS="${PHASE2B_SEEDS}" \
STEPS="${PHASE2B_STEPS}" \
EVAL_EXAMPLES="${PHASE2B_EVAL_EXAMPLES}" \
"${SCRIPT_DIR}/with_conda_npu.sh" "${SCRIPT_DIR}/run_phase2b_pointer_matrix.sh"
printf '[%s] phase2b pointer matrix done\n' "$(date -Is)"

printf '[%s] phase2a ladder scale start\n' "$(date -Is)"
DEVICE="${DEVICE}" \
DATA_DIR="${PHASE2A_DATA_DIR}" \
OUTPUT_DIR="${PHASE2A_OUTPUT_DIR}" \
SEEDS="${PHASE2A_SEEDS}" \
STEPS="${PHASE2A_STEPS}" \
EVAL_EXAMPLES="${PHASE2A_EVAL_EXAMPLES}" \
"${SCRIPT_DIR}/with_conda_npu.sh" "${SCRIPT_DIR}/run_phase2a_ladder_scale.sh"
printf '[%s] phase2a ladder scale done\n' "$(date -Is)"
