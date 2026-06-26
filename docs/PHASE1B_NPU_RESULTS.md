# Phase 1b NPU Results

Date: 2026-06-26

Note: the five-method matrix in this file used mixed evaluation modes: binary-choice scoring for `direct/soft/latent`, and generation parsing for `cot/masked_cot`. It is useful as a runability check. The comparable final-answer protocol is documented in `docs/PHASE1C_EVAL_AND_DIAGNOSTICS.md`.

Environment:

- Wrapper: `scripts/with_conda_npu.sh`
- Toolkit path: `/usr/local/Ascend/ascend-toolkit/8.2.RC1`
- Python: `/home/zlong/anaconda3/envs/clt-npu-py39/bin/python`
- Device: `ASCEND_RT_VISIBLE_DEVICES=5`, `DEVICE=npu:0`
- Task: `graph_reachability`
- Difficulty: `easy`
- Model: `d_model=32`, `n_layers=1`, `n_heads=2`

## Direct Baseline

Command:

```bash
DEVICE=npu:0 \
DATA_DIR=data/phase1b_easy_npu_conda \
OUTPUT_DIR=outputs/phase1b_direct_curve_npu_conda \
STEPS_LIST='100 300 1000 3000' \
EVAL_EXAMPLES=200 \
scripts/with_conda_npu.sh \
scripts/run_phase1b_direct_curve.sh
```

Results:

| steps | dev | id_test | ood_test | loss | sec |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.565 | 0.515 | 0.435 | 2.797 | 0.9 |
| 300 | 0.585 | 0.635 | 0.450 | 0.559 | 2.1 |
| 1000 | 0.535 | 0.590 | 0.485 | 0.082 | 5.5 |
| 3000 | 1.000 | 1.000 | 0.470 | 0.012 | 13.1 |

Observation: the direct model can fit the ID easy task by 3000 steps, but OOD graph-size generalization remains near chance.

## Five-Method Matrix

Command:

```bash
DEVICE=npu:0 \
DATA_DIR=data/phase1b_matrix_npu_conda \
OUTPUT_DIR=outputs/phase1b_method_matrix_npu_conda \
STEPS_LIST='100 300 1000' \
EVAL_EXAMPLES=50 \
K=8 \
MAX_NEW_TOKENS_GENERATE=96 \
scripts/with_conda_npu.sh \
scripts/run_phase1b_method_matrix.sh
```

Evaluation modes:

- `direct`, `soft`, and `latent`: binary-choice scoring over `YES` vs `NO`.
- `cot` and `masked_cot`: generation from `Problem -> Reasoning:` followed by answer parsing.

Results:

| method | steps | eval | k | dev | id_test | ood_test | loss | sec |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| direct | 100 | binary_choice |  | 0.580 | 0.500 | 0.460 | 2.797 | 0.9 |
| direct | 300 | binary_choice |  | 0.560 | 0.640 | 0.440 | 0.559 | 1.8 |
| direct | 1000 | binary_choice |  | 0.560 | 0.620 | 0.480 | 0.082 | 4.7 |
| cot | 100 | generate |  | 0.000 | 0.000 | 0.000 | 2.967 | 0.8 |
| cot | 300 | generate |  | 0.000 | 0.000 | 0.000 | 1.436 | 2.0 |
| cot | 1000 | generate |  | 0.580 | 0.500 | 0.700 | 0.489 | 5.5 |
| masked_cot | 100 | generate |  | 0.000 | 0.000 | 0.120 | 2.479 | 0.9 |
| masked_cot | 300 | generate |  | 0.140 | 0.120 | 0.780 | 0.478 | 1.9 |
| masked_cot | 1000 | generate |  | 0.440 | 0.380 | 0.460 | 0.050 | 4.2 |
| soft | 100 | binary_choice | 8 | 0.580 | 0.500 | 0.500 | 2.623 | 3.1 |
| soft | 300 | binary_choice | 8 | 0.560 | 0.640 | 0.340 | 0.524 | 7.9 |
| soft | 1000 | binary_choice | 8 | 0.560 | 0.620 | 0.440 | 0.074 | 21.1 |
| latent | 100 | binary_choice | 8 | 0.580 | 0.500 | 0.500 | 2.598 | 3.0 |
| latent | 300 | binary_choice | 8 | 0.540 | 0.620 | 0.420 | 0.496 | 8.5 |
| latent | 1000 | binary_choice | 8 | 0.560 | 0.620 | 0.320 | 0.070 | 24.8 |

## Initial Takeaways

1. The five training paths all run end to end on the Ascend NPU environment.
2. Direct, soft, and latent have very similar ID behavior at 1000 steps under binary-choice scoring.
3. Continuous methods are slower here because each example performs `K=8` recurrent continuous steps before answer scoring.
4. CoT-style methods need a better comparable evaluation protocol. Generation can become parseable by 1000 steps, but it is not directly comparable to binary-choice scoring.
5. OOD accuracy is unstable in this first single-seed run. The next useful experiment is a multi-seed run plus a common verifier/scoring setup for all methods.
