#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-graph_reachability}"
DIFFICULTY="${DIFFICULTY:-easy}"
DATA_DIR="${DATA_DIR:-data/phase1b_k_sweep}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase1b_k_sweep}"
DEVICE="${DEVICE:-cpu}"
METHODS="${METHODS:-soft latent}"
K_LIST="${K_LIST:-0 2 4 8 16}"
SEED="${SEED:-0}"
STEPS="${STEPS:-1000}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-50}"
D_MODEL="${D_MODEL:-32}"
N_LAYERS="${N_LAYERS:-1}"
N_HEADS="${N_HEADS:-2}"
LR="${LR:-0.0003}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
PYTHON="${PYTHON:-python3}"

mkdir -p "${OUTPUT_DIR}"

PYTHONPATH="src:${PYTHONPATH:-}" "${PYTHON}" -m clt.build_dataset \
  --task "${TASK}" \
  --preset debug \
  --difficulty "${DIFFICULTY}" \
  --out-dir "${DATA_DIR}"

for method in ${METHODS}; do
  for k in ${K_LIST}; do
    echo "Running K sweep point: method=${method} k=${k} steps=${STEPS}"
    PYTHONUNBUFFERED=1 PYTHONPATH="src:${PYTHONPATH:-}" "${PYTHON}" -m clt.train_tiny \
      --task "${TASK}" \
      --method "${method}" \
      --difficulty "${DIFFICULTY}" \
      --data-dir "${DATA_DIR}" \
      --device "${DEVICE}" \
      --steps "${STEPS}" \
      --eval-examples "${EVAL_EXAMPLES}" \
      --eval-mode binary_choice \
      --lr "${LR}" \
      --k "${k}" \
      --seed "${SEED}" \
      --d-model "${D_MODEL}" \
      --n-layers "${N_LAYERS}" \
      --n-heads "${N_HEADS}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --output "${OUTPUT_DIR}/${method}_k${k}.json" \
      2>&1 | tee "${OUTPUT_DIR}/${method}_k${k}.log"
  done
done

OUTPUT_DIR="${OUTPUT_DIR}" METHODS="${METHODS}" PYTHONPATH="src:${PYTHONPATH:-}" "${PYTHON}" - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_DIR"])
method_order = {method: i for i, method in enumerate(os.environ["METHODS"].split())}
rows = []
for path in root.glob("*_k*.json"):
    method, k_text = path.stem.rsplit("_k", 1)
    payload = json.loads(path.read_text())
    rows.append(
        {
            "method": method,
            "k": int(k_text),
            "steps": payload["steps"],
            "dev": payload["dev"]["accuracy"],
            "id_test": payload["id_test"]["accuracy"],
            "ood_test": payload["ood_test"]["accuracy"],
            "loss": payload["train_loss_last"],
            "elapsed_sec": payload["elapsed_sec"],
        }
    )
rows.sort(key=lambda row: (method_order.get(row["method"], 999), row["k"]))

print("\nPhase 1b K sweep")
print("method\tk\tid_test\tood_test\tloss\tsec")
for row in rows:
    print(
        f"{row['method']}\t"
        f"{row['k']}\t"
        f"{row['id_test']:.3f}\t"
        f"{row['ood_test']:.3f}\t"
        f"{row['loss']:.3f}\t"
        f"{row['elapsed_sec']:.1f}"
    )

(root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
    if rows:
        writer.writeheader()
        writer.writerows(rows)
PY
