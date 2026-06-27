# Phase 1c Evaluation and Diagnostics

Date: 2026-06-26

Environment:

- Wrapper: `scripts/with_conda_npu.sh`
- Python: `/home/zlong/anaconda3/envs/fdt-npu-py39/bin/python`
- Toolkit path: `/usr/local/Ascend/ascend-toolkit/8.2.RC1`
- Device: `ASCEND_RT_VISIBLE_DEVICES=5`, `DEVICE=npu:0`
- Task: `graph_reachability`
- Difficulty: `easy`
- Model: `d_model=32`, `n_layers=1`, `n_heads=2`

## Evaluation Protocol

All five methods now support `--eval-mode binary_choice`.

- `direct`: score `YES\n` and `NO\n` after `Problem -> Answer:`.
- `soft` / `latent`: run `K` continuous steps, then score `YES\n` and `NO\n`.
- `cot` / `masked_cot`: generate a self-produced reasoning prefix from `Problem -> Reasoning:`, then score `YES\n` and `NO\n` after `Answer:`.

The CoT variants do not receive oracle traces at test time. This makes final-answer scoring comparable across methods while preserving the fact that CoT-style methods spend test-time compute on textual reasoning.

## Multi-Seed Matrix

Command:

```bash
DEVICE=npu:0 \
DATA_DIR=data/phase1b_multiseed_npu_choice \
OUTPUT_DIR=outputs/phase1b_multiseed_npu_choice \
SEEDS='0 1 2' \
STEPS=1000 \
EVAL_EXAMPLES=50 \
K=8 \
MAX_NEW_TOKENS=96 \
scripts/with_conda_npu.sh \
scripts/run_phase1b_multiseed_matrix.sh
```

Aggregate results:

| method | n | id_mean | id_std | ood_mean | ood_std | loss_mean | sec_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct | 3 | 0.580 | 0.069 | 0.527 | 0.064 | 0.182 | 3.5 |
| cot | 3 | 0.500 | 0.000 | 0.580 | 0.104 | 0.305 | 3.6 |
| masked_cot | 3 | 0.553 | 0.076 | 0.460 | 0.106 | 0.045 | 3.5 |
| soft | 3 | 0.580 | 0.069 | 0.493 | 0.061 | 0.187 | 20.0 |
| latent | 3 | 0.580 | 0.069 | 0.453 | 0.122 | 0.185 | 19.0 |

Initial read:

- Under the unified binary-choice protocol, direct/soft/latent have nearly identical ID behavior in this small setup.
- `soft` and `latent` are much slower at `K=8` without ID gains.
- CoT's OOD mean is highest in this 3-seed run, but variance is high and the textual trace policy is noisy.

## K Sweep

Command:

```bash
DEVICE=npu:0 \
DATA_DIR=data/phase1b_k_sweep_npu \
OUTPUT_DIR=outputs/phase1b_k_sweep_npu \
K_LIST='0 2 4 8 16' \
STEPS=1000 \
EVAL_EXAMPLES=50 \
SEED=0 \
scripts/with_conda_npu.sh \
scripts/run_phase1b_k_sweep.sh
```

Results:

| method | K | id_test | ood_test | loss | sec |
|---|---:|---:|---:|---:|---:|
| soft | 0 | 0.620 | 0.480 | 0.082 | 3.4 |
| soft | 2 | 0.620 | 0.260 | 0.067 | 7.2 |
| soft | 4 | 0.620 | 0.240 | 0.067 | 11.2 |
| soft | 8 | 0.620 | 0.440 | 0.074 | 19.3 |
| soft | 16 | 0.620 | 0.540 | 0.073 | 34.7 |
| latent | 0 | 0.620 | 0.480 | 0.082 | 3.6 |
| latent | 2 | 0.620 | 0.240 | 0.062 | 7.7 |
| latent | 4 | 0.620 | 0.340 | 0.062 | 11.3 |
| latent | 8 | 0.620 | 0.320 | 0.070 | 18.9 |
| latent | 16 | 0.620 | 0.540 | 0.065 | 33.9 |

Initial read:

- ID accuracy is flat across `K`.
- Larger `K` increases runtime roughly linearly.
- The OOD variation in this single-seed sweep is not enough to claim a K benefit; it should be repeated under multi-seed if it becomes important.

## OOD Node Ladder

Command:

```bash
DEVICE=npu:0 \
DATA_DIR=data/phase1b_ood_diag_npu \
OUTPUT_DIR=outputs/phase1b_ood_diag_npu \
METHOD=direct \
STEPS=3000 \
EVAL_EXAMPLES=100 \
DIAGNOSTIC_EXAMPLES=100 \
DIAGNOSTIC_NODES='4,5,6,7,8' \
scripts/with_conda_npu.sh \
scripts/run_phase1b_ood_diagnosis.sh
```

Results:

| split | accuracy | n |
|---|---:|---:|
| dev | 1.000 | 100 |
| id_test | 1.000 | 100 |
| ood_test | 0.430 | 100 |
| easy_n4 | 0.800 | 100 |
| easy_n5 | 0.470 | 100 |
| easy_n6 | 0.570 | 100 |
| easy_n7 | 0.500 | 100 |
| easy_n8 | 0.450 | 100 |

Initial read:

- The direct model fits the original ID distribution, but performance drops sharply as soon as node count differs from the training setting.
- The task currently has a strong distribution shift around graph size and edge-pattern format, so OOD claims should be treated as diagnostic rather than as a core method comparison.

## Next Decisions

1. Add checkpointing so the same trained model can be evaluated on many diagnostics without retraining.
2. Add a smoother graph-size curriculum, for example training on `n=4..6` and testing on `n=7..8`.
3. Repeat the K sweep with 3 seeds only if the project wants to investigate OOD sensitivity; current evidence does not show an ID benefit from continuous thinking.
