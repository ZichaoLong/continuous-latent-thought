# Phase 1b Tiny Learnability Check

Phase 1b checks whether the tiny setup can learn the easiest graph task before comparing Direct, CoT, Soft Token, and Latent Thought.

The target is modest:

```text
Direct Answer on easy graph_reachability should exceed random accuracy under binary-choice evaluation.
```

## What Changed

Phase 1b adds:

- `difficulty=easy` for `graph_reachability`;
- binary candidate scoring for `YES` vs `NO`;
- a Direct Answer learning-curve runner.

Binary-choice evaluation scores:

```text
P("YES" | prompt)
P("NO"  | prompt)
```

and selects the answer with lower average token NLL. This avoids free-generation noise from the tiny character model.

## Build Easy Data

```bash
PYTHONPATH=src python3 -m clt.build_dataset \
  --task graph_reachability \
  --preset debug \
  --difficulty easy \
  --out-dir data/phase1b_easy
```

The `debug` preset writes:

```text
train:    2000 examples
dev:       200 examples
id_test:   200 examples
ood_test:  200 examples
```

## Run One Direct Experiment

```bash
PYTHONPATH=src python3 -m clt.train_tiny \
  --task graph_reachability \
  --method direct \
  --difficulty easy \
  --data-dir data/phase1b_easy \
  --device cpu \
  --steps 300 \
  --eval-examples 200 \
  --eval-mode binary_choice \
  --d-model 64 \
  --n-layers 2 \
  --n-heads 4
```

## Run Direct Learning Curve

Foreground:

```bash
scripts/run_phase1b_direct_curve.sh
```

Background:

```bash
mkdir -p outputs/phase1b_direct_curve
nohup bash -lc 'scripts/run_phase1b_direct_curve.sh' \
  > outputs/phase1b_direct_curve/background.log 2>&1 &
echo $! > outputs/phase1b_direct_curve/pid
```

Check progress:

```bash
tail -f outputs/phase1b_direct_curve/background.log
cat outputs/phase1b_direct_curve/pid
```

Check results:

```bash
cat outputs/phase1b_direct_curve/summary.json
ls outputs/phase1b_direct_curve/direct_*.json
```

Default curve points:

```text
100, 300, 1000 steps
```

Override with:

```bash
STEPS_LIST="100 300 1000 3000" scripts/run_phase1b_direct_curve.sh
```

## Interpretation

If Direct Answer cannot beat random on this easy setting, the next work should focus on:

- model capacity;
- optimizer or learning rate;
- answer scoring;
- task formatting.

Only after Direct Answer learns should we compare:

```text
Direct
Standard CoT
Soft Token K=4/8
Latent Thought K=4/8
```
