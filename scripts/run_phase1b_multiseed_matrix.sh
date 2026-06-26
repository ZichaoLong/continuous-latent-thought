#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-graph_reachability}"
DIFFICULTY="${DIFFICULTY:-easy}"
DATA_DIR="${DATA_DIR:-data/phase1b_multiseed}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase1b_multiseed_matrix}"
DEVICE="${DEVICE:-cpu}"
METHODS="${METHODS:-direct cot masked_cot soft latent}"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-1000}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-50}"
K="${K:-8}"
D_MODEL="${D_MODEL:-32}"
N_LAYERS="${N_LAYERS:-1}"
N_HEADS="${N_HEADS:-2}"
LR="${LR:-0.0003}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
PYTHON="${PYTHON:-python3}"

mkdir -p "${OUTPUT_DIR}"

PYTHONPATH="src:${PYTHONPATH:-}" "${PYTHON}" -m clt.build_dataset \
  --task "${TASK}" \
  --preset debug \
  --difficulty "${DIFFICULTY}" \
  --out-dir "${DATA_DIR}"

for seed in ${SEEDS}; do
  for method in ${METHODS}; do
    echo "Running multiseed point: method=${method} seed=${seed} steps=${STEPS}"
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
      --k "${K}" \
      --seed "${seed}" \
      --d-model "${D_MODEL}" \
      --n-layers "${N_LAYERS}" \
      --n-heads "${N_HEADS}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --output "${OUTPUT_DIR}/${method}_seed${seed}.json" \
      2>&1 | tee "${OUTPUT_DIR}/${method}_seed${seed}.log"
  done
done

OUTPUT_DIR="${OUTPUT_DIR}" METHODS="${METHODS}" PYTHONPATH="src:${PYTHONPATH:-}" "${PYTHON}" - <<'PY'
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

root = Path(os.environ["OUTPUT_DIR"])
method_order = {method: i for i, method in enumerate(os.environ["METHODS"].split())}

rows = []
for path in root.glob("*_seed*.json"):
    payload = json.loads(path.read_text())
    stem = path.stem
    method, seed_text = stem.rsplit("_seed", 1)
    rows.append(
        {
            "method": method,
            "seed": int(seed_text),
            "steps": payload["steps"],
            "eval_mode": payload["eval_mode"],
            "k": "" if payload["k"] is None else payload["k"],
            "dev": payload["dev"]["accuracy"],
            "id_test": payload["id_test"]["accuracy"],
            "ood_test": payload["ood_test"]["accuracy"],
            "loss": payload["train_loss_last"],
            "elapsed_sec": payload["elapsed_sec"],
        }
    )

rows.sort(key=lambda row: (method_order.get(row["method"], 999), row["seed"]))

def mean(values):
    return sum(values) / len(values)

def std(values):
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))

groups = defaultdict(list)
for row in rows:
    groups[(row["method"], row["steps"], row["k"], row["eval_mode"])].append(row)

aggregate = []
for (method, steps, k, eval_mode), group in groups.items():
    aggregate.append(
        {
            "method": method,
            "steps": steps,
            "eval_mode": eval_mode,
            "k": k,
            "n": len(group),
            "dev_mean": mean([row["dev"] for row in group]),
            "dev_std": std([row["dev"] for row in group]),
            "id_test_mean": mean([row["id_test"] for row in group]),
            "id_test_std": std([row["id_test"] for row in group]),
            "ood_test_mean": mean([row["ood_test"] for row in group]),
            "ood_test_std": std([row["ood_test"] for row in group]),
            "loss_mean": mean([row["loss"] for row in group]),
            "elapsed_sec_mean": mean([row["elapsed_sec"] for row in group]),
        }
    )
aggregate.sort(key=lambda row: method_order.get(row["method"], 999))

print("\nPhase 1b multiseed aggregate")
print("method\tn\tid_mean\tid_std\tood_mean\tood_std\tloss_mean\tsec_mean")
for row in aggregate:
    print(
        f"{row['method']}\t"
        f"{row['n']}\t"
        f"{row['id_test_mean']:.3f}\t"
        f"{row['id_test_std']:.3f}\t"
        f"{row['ood_test_mean']:.3f}\t"
        f"{row['ood_test_std']:.3f}\t"
        f"{row['loss_mean']:.3f}\t"
        f"{row['elapsed_sec_mean']:.1f}"
    )

(root / "summary.json").write_text(json.dumps({"runs": rows, "aggregate": aggregate}, indent=2), encoding="utf-8")
for name, payload in [("runs.csv", rows), ("aggregate.csv", aggregate)]:
    with (root / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload[0].keys()) if payload else [])
        if payload:
            writer.writeheader()
            writer.writerows(payload)
PY
