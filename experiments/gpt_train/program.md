# Experiment: gpt_train

You are optimizing a small GPT language model trained on a **single Intel Gaudi 2
HPU**. The artifact you edit is `train.py` (model architecture, optimizer, and
training loop). Every run uses a **fixed 5-minute training budget**, so you never
trade quality for speed — only better-quality-per-fixed-budget matters.

## Goal

**Minimize `val_bpb`** (validation bits-per-byte). Lower is better. The harness
runs the fixed scorer for you; you only ever see the measured metric.

## What you CAN change (in `train.py`)

Everything is fair game: model architecture, depth/width, optimizer and its
hyperparameters, learning rate and schedule, batch size, initialization, the
training loop. Make ONE coherent change per iteration so cause and effect are clear.

## What you CANNOT do

- You cannot modify the evaluation, data loading, tokenizer, or fixed constants
  (those live in `prepare.py`, which is read-only and not exposed to you).
- You cannot add dependencies or install packages — use only what `train.py`
  already imports.
- You cannot change the device or graph-mode setup. This runs in **HPU lazy graph
  mode**. NEVER switch to eager mode and NEVER remove `htcore.mark_step()` calls —
  doing so makes the run catastrophically slow and is treated as a failure.

## Constraints

- **VRAM** is a soft constraint. Modest increases are fine for real `val_bpb`
  gains, but do not blow it up dramatically (risk of OOM crash → discarded).
- **Simplicity**: all else equal, simpler is better. A tiny gain that adds ugly
  complexity is not worth it; an equal-or-better result with *less* code is a win.
- The code must run without crashing and finish within the time budget.

## What a run reports

`run_experiment` runs the fixed scorer and returns the measured metrics, e.g.:

```
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

`val_bpb` is the number you optimize. The rest are context: `peak_vram_mb` for the
VRAM constraint, `mfu_percent` / `num_steps` / `total_tokens_M` for how much
training fit in the fixed budget. The harness records every run to `results.tsv`
for you — you never edit that file yourself.

## How to decide keep vs discard

- After `run_experiment`, compare the measured `val_bpb` against the current best.
- If it is **lower** (improved), call `keep` with a short description.
- If it is **equal, worse, or crashed**, call `discard` with the reason. The
  harness reverts the artifact to the previous best automatically.

You MUST end every iteration with exactly one `keep` or `discard`.

## Mindset

You are an autonomous researcher: form a hypothesis, make one coherent change,
measure it, and let the result decide. Keep what works, discard what doesn't, and
keep iterating. If a run crashes on something trivial (a typo, a bad tensor shape),
fix it and re-run; if the idea itself is fundamentally broken, discard it and move
on. When you run low on ideas, look harder — revisit the papers referenced in the
code, re-read `train.py` for new angles, combine previous near-misses, or try a
more radical architectural change.
