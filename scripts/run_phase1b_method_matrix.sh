#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-graph_reachability}"
DIFFICULTY="${DIFFICULTY:-easy}"
DATA_DIR="${DATA_DIR:-data/phase1b_easy_matrix}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase1b_method_matrix}"
DEVICE="${DEVICE:-cpu}"
METHODS="${METHODS:-direct cot masked_cot soft latent}"
STEPS_LIST="${STEPS_LIST:-100 300 1000}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-100}"
K="${K:-8}"
D_MODEL="${D_MODEL:-32}"
N_LAYERS="${N_LAYERS:-1}"
N_HEADS="${N_HEADS:-2}"
LR="${LR:-0.0003}"
PYTHON="${PYTHON:-python3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"

mkdir -p "${OUTPUT_DIR}"

PYTHONPATH="src:${PYTHONPATH:-}" "${PYTHON}" -m fdt.build_dataset \
  --task "${TASK}" \
  --preset debug \
  --difficulty "${DIFFICULTY}" \
  --out-dir "${DATA_DIR}"

eval_mode_for_method() {
  case "$1" in
    direct|cot|masked_cot|soft|latent) echo "binary_choice" ;;
    *) echo "Unknown method: $1" >&2; exit 2 ;;
  esac
}

for method in ${METHODS}; do
  eval_mode="$(eval_mode_for_method "${method}")"
  for steps in ${STEPS_LIST}; do
    echo "Running matrix point: method=${method} steps=${steps} eval_mode=${eval_mode}"
    PYTHONUNBUFFERED=1 PYTHONPATH="src:${PYTHONPATH:-}" "${PYTHON}" -m fdt.train_tiny \
      --task "${TASK}" \
      --method "${method}" \
      --difficulty "${DIFFICULTY}" \
      --data-dir "${DATA_DIR}" \
      --device "${DEVICE}" \
      --steps "${steps}" \
      --eval-examples "${EVAL_EXAMPLES}" \
      --eval-mode "${eval_mode}" \
      --lr "${LR}" \
      --k "${K}" \
      --d-model "${D_MODEL}" \
      --n-layers "${N_LAYERS}" \
      --n-heads "${N_HEADS}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --output "${OUTPUT_DIR}/${method}_${steps}.json" \
      2>&1 | tee "${OUTPUT_DIR}/${method}_${steps}.log"
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
for path in root.glob("*.json"):
    if path.name in {"summary.json"}:
        continue
    method, steps_text = path.stem.rsplit("_", 1)
    payload = json.loads(path.read_text())
    rows.append(
        {
            "method": method,
            "steps": int(steps_text),
            "eval_mode": payload["eval_mode"],
            "k": "" if payload["k"] is None else payload["k"],
            "dev": payload["dev"]["accuracy"],
            "id_test": payload["id_test"]["accuracy"],
            "ood_test": payload["ood_test"]["accuracy"],
            "loss": payload["train_loss_last"],
            "elapsed_sec": payload["elapsed_sec"],
        }
    )

rows.sort(key=lambda row: (method_order.get(row["method"], 999), row["steps"]))

print("\nPhase 1b method matrix")
print("method\tsteps\teval\tk\tdev\tid_test\tood_test\tloss\tsec")
for row in rows:
    print(
        f"{row['method']}\t"
        f"{row['steps']}\t"
        f"{row['eval_mode']}\t"
        f"{row['k']}\t"
        f"{row['dev']:.3f}\t"
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
