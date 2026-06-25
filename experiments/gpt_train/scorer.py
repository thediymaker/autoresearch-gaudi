#!/usr/bin/env python3
"""Fixed scorer for the gpt_train experiment.

Invoked by the harness as:  python scorer.py <path-to-train.py>

It copies the candidate train.py into the autoresearch debug pod, runs one
training experiment (fixed 5-minute budget inside the script), parses the
ground-truth metrics from the run log, and prints a single JSON object to
stdout:

  {"status": "ok", "metrics": {"val_bpb": <float>, "peak_vram_mb": <float>,
                               "num_steps": <int>}}

or, on failure:

  {"status": "crash", "metrics": {}, "error": "<msg>", "log_tail": "<...>"}

This file is FIXED harness code — the agent never edits it. It is the only
component allowed to touch the cluster, and only via the narrow kubectl
operations below (copy file, run script, read log) in the `autoresearch`
namespace.
"""
from __future__ import annotations

import json
import subprocess
import sys

NAMESPACE = "autoresearch"
POD = "autoresearch-debug"
POD_TRAIN_PATH = "/workspace/autoresearch/train.py"
LOG_PATH = "/root/.cache/autoresearch/run.log"
RUN_TIMEOUT = 720  # 12 min hard ceiling (5-min budget + startup/compile/eval)


def _run(cmd, timeout):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _log_tail(n=50):
    try:
        r = _run(["kubectl", "exec", "-n", NAMESPACE, POD, "--",
                  "tail", "-n", str(n), LOG_PATH], timeout=60)
        return r.stdout[-2000:]
    except Exception as e:  # noqa: BLE001
        return f"(could not read log: {e})"


def crash(msg, include_log=True):
    out = {"status": "crash", "metrics": {}, "error": msg}
    if include_log:
        out["log_tail"] = _log_tail()
    print(json.dumps(out))
    sys.exit(0)


def main():
    if len(sys.argv) < 2:
        crash("usage: scorer.py <path-to-train.py>", include_log=False)
    artifact = sys.argv[1]

    # 1) Pod must be Running.
    try:
        r = _run(["kubectl", "get", "pod", POD, "-n", NAMESPACE,
                  "-o", "jsonpath={.status.phase}"], timeout=60)
    except subprocess.TimeoutExpired:
        crash("kubectl get pod timed out", include_log=False)
    if r.stdout.strip() != "Running":
        crash(f"debug pod not Running (phase={r.stdout.strip()!r})", include_log=False)

    # 2) Copy candidate train.py into the pod.
    try:
        r = _run(["kubectl", "cp", artifact, f"{NAMESPACE}/{POD}:{POD_TRAIN_PATH}"], timeout=120)
    except subprocess.TimeoutExpired:
        crash("kubectl cp timed out", include_log=False)
    if r.returncode != 0:
        crash(f"kubectl cp failed: {r.stderr.strip()}", include_log=False)

    # 3) Run the experiment (blocks ~5 min + startup).
    try:
        _run(["kubectl", "exec", "-n", NAMESPACE, POD, "--", "bash", "-c",
              f"cd /workspace/autoresearch && python train.py > {LOG_PATH} 2>&1"],
             timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        crash("training run exceeded timeout")

    # 4) Parse metrics from the log.
    try:
        r = _run(["kubectl", "exec", "-n", NAMESPACE, POD, "--",
                  "grep", "-E", "^(val_bpb|peak_vram_mb|num_steps):", LOG_PATH], timeout=60)
    except subprocess.TimeoutExpired:
        crash("grep on log timed out")

    metrics = {}
    for line in r.stdout.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        try:
            if key == "val_bpb":
                metrics["val_bpb"] = float(val)
            elif key == "peak_vram_mb":
                metrics["peak_vram_mb"] = float(val)
            elif key == "num_steps":
                metrics["num_steps"] = int(float(val))
        except ValueError:
            continue

    if "val_bpb" not in metrics:
        crash("no val_bpb in log (run likely crashed)")

    print(json.dumps({"status": "ok", "metrics": metrics}))


if __name__ == "__main__":
    main()
