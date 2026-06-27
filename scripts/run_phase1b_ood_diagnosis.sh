#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-graph_reachability}"
DIFFICULTY="${DIFFICULTY:-easy}"
DATA_DIR="${DATA_DIR:-data/phase1b_ood_diag}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase1b_ood_diagnosis}"
DEVICE="${DEVICE:-cpu}"
METHOD="${METHOD:-direct}"
SEED="${SEED:-0}"
STEPS="${STEPS:-3000}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-100}"
DIAGNOSTIC_EXAMPLES="${DIAGNOSTIC_EXAMPLES:-100}"
DIAGNOSTIC_NODES="${DIAGNOSTIC_NODES:-4,5,6,7,8}"
K="${K:-8}"
D_MODEL="${D_MODEL:-32}"
N_LAYERS="${N_LAYERS:-1}"
N_HEADS="${N_HEADS:-2}"
LR="${LR:-0.0003}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
PYTHON="${PYTHON:-python3}"

mkdir -p "${OUTPUT_DIR}"

PYTHONPATH="src:${PYTHONPATH:-}" "${PYTHON}" -m fdt.build_dataset \
  --task "${TASK}" \
  --preset debug \
  --difficulty "${DIFFICULTY}" \
  --out-dir "${DATA_DIR}"

PYTHONUNBUFFERED=1 PYTHONPATH="src:${PYTHONPATH:-}" "${PYTHON}" -m fdt.train_tiny \
  --task "${TASK}" \
  --method "${METHOD}" \
  --difficulty "${DIFFICULTY}" \
  --data-dir "${DATA_DIR}" \
  --device "${DEVICE}" \
  --steps "${STEPS}" \
  --eval-examples "${EVAL_EXAMPLES}" \
  --eval-mode binary_choice \
  --lr "${LR}" \
  --k "${K}" \
  --seed "${SEED}" \
  --d-model "${D_MODEL}" \
  --n-layers "${N_LAYERS}" \
  --n-heads "${N_HEADS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --easy-graph-diagnostic-nodes "${DIAGNOSTIC_NODES}" \
  --diagnostic-examples "${DIAGNOSTIC_EXAMPLES}" \
  --output "${OUTPUT_DIR}/${METHOD}_diagnosis.json" \
  2>&1 | tee "${OUTPUT_DIR}/${METHOD}_diagnosis.log"

OUTPUT_DIR="${OUTPUT_DIR}" METHOD="${METHOD}" PYTHONPATH="src:${PYTHONPATH:-}" "${PYTHON}" - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_DIR"])
method = os.environ["METHOD"]
payload = json.loads((root / f"{method}_diagnosis.json").read_text())
rows = [
    {"split": "dev", "accuracy": payload["dev"]["accuracy"], "num_examples": payload["dev"]["num_examples"]},
    {"split": "id_test", "accuracy": payload["id_test"]["accuracy"], "num_examples": payload["id_test"]["num_examples"]},
    {"split": "ood_test", "accuracy": payload["ood_test"]["accuracy"], "num_examples": payload["ood_test"]["num_examples"]},
]
for name, metric in payload.get("diagnostics", {}).items():
    rows.append({"split": name, "accuracy": metric["accuracy"], "num_examples": metric["num_examples"]})

print("\nPhase 1b OOD diagnosis")
print("split\taccuracy\tn")
for row in rows:
    print(f"{row['split']}\t{row['accuracy']:.3f}\t{row['num_examples']}")

(root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
    if rows:
        writer.writeheader()
        writer.writerows(rows)
PY
