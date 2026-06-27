# Phase 0 Execution Notes

Phase 0 builds the minimal infrastructure needed before LLM SFT experiments:

- deterministic synthetic task generators;
- JSONL data export;
- answer verification;
- unit tests;
- Ascend NPU smoke test.

## Run Tests

```bash
PYTHONPATH=src pytest -q
```

## Generate Debug Data

```bash
PYTHONPATH=src python3 -m fdt.generate_data \
  --task all \
  --split train \
  --num-examples 8 \
  --seed-start 0 \
  --out-dir data/debug
```

Generated files are ignored by git under `data/`.

## Ascend NPU Smoke Test

This machine has Ascend910 NPUs. The default shell environment points at CANN 8.5.1, but the installed `torch_npu==2.7.1rc1` works with the 8.2 toolkit paths in this environment.

Use the wrapper:

```bash
ASCEND_RT_VISIBLE_DEVICES=5 scripts/with_ascend_env.sh \
  python3 scripts/npu_smoke.py --device npu:0 --size 128
```

`ASCEND_RT_VISIBLE_DEVICES=5` restricts the process to one currently idle physical NPU. Inside the process, it appears as logical `npu:0`.

Python packages needed by the local NPU stack:

```bash
python3 -m pip install --user PyYAML attrs
```

These are runtime environment requirements for this machine, not project data dependencies.
