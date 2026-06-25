#!/usr/bin/env python3
"""Fixed scorer for the gpt_train experiment.

Invoked by the harness as:  python scorer.py <path-to-train.py>

Runs one training experiment (fixed 5-minute budget inside train.py), parses the
ground-truth metrics from its output, and prints a single JSON object to stdout:

  {"status": "ok", "metrics": {"val_bpb": <float>, "peak_vram_mb": <float>,
                               "num_steps": <int>}}

or, on failure:

  {"status": "crash", "metrics": {}, "error": "<msg>", "log_tail": "<...>"}

There are two execution backends, chosen by the AUTORESEARCH_RUNNER env var:

  - "local" : run train.py directly on THIS machine. Use this when the harness
              runs ON the Gaudi node / inside the Gaudi pod (e.g. the JupyterLab
              template, where the repo, the notebook, and the HPU are all
              co-located). No kubectl, no copying — the artifact is run in place.
  - "k8s"   : kubectl cp the artifact into a separate debug pod, run it there,
              and read the log back. Use this when the harness runs outside the
              cluster and drives a remote debug pod.
  - unset   : auto-detect — "k8s" if kubectl is installed AND the debug pod is
              Running, otherwise "local".

This file is FIXED harness code — the agent never edits it. In k8s mode it is the
only component allowed to touch the cluster, and only via the narrow kubectl
operations below (copy file, run script, read log).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

# --- k8s backend config (overridable; matches autoresearch-debug.yaml) ------
NAMESPACE = os.environ.get("AUTORESEARCH_NAMESPACE", "autoresearch")
POD = os.environ.get("AUTORESEARCH_POD", "autoresearch-debug")
POD_TRAIN_PATH = "/workspace/autoresearch/train.py"
LOG_PATH = "/root/.cache/autoresearch/run.log"

RUN_TIMEOUT = 720  # 12 min hard ceiling (5-min budget + startup/compile/eval)


def _run(cmd, timeout):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def crash(msg, log=None):
    out = {"status": "crash", "metrics": {}, "error": msg}
    if log:
        out["log_tail"] = log[-2000:]
    print(json.dumps(out))
    sys.exit(0)


def parse_metrics(text: str) -> dict:
    """Pull val_bpb / peak_vram_mb / num_steps from train.py's summary output.

    train.py prints lines like 'val_bpb:          0.997900'. We match on the key
    at the start of a (stripped) line so stray mid-sentence mentions are ignored.
    """
    metrics = {}
    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        try:
            if key == "val_bpb":
                metrics["val_bpb"] = float(val)
            elif key == "peak_vram_mb":
                metrics["peak_vram_mb"] = float(val)
            elif key == "num_steps":
                metrics["num_steps"] = int(float(val))
        except ValueError:
            continue
    return metrics


# --------------------------------------------------------------------------- #
# Runner selection
# --------------------------------------------------------------------------- #
def detect_runner() -> str:
    choice = os.environ.get("AUTORESEARCH_RUNNER", "").strip().lower()
    if choice in ("local", "k8s"):
        return choice
    # Auto: prefer k8s only if kubectl exists AND the debug pod is Running.
    if shutil.which("kubectl"):
        try:
            r = _run(["kubectl", "get", "pod", POD, "-n", NAMESPACE,
                      "-o", "jsonpath={.status.phase}"], timeout=30)
            if r.returncode == 0 and r.stdout.strip() == "Running":
                return "k8s"
        except Exception:  # noqa: BLE001
            pass
    return "local"


# --------------------------------------------------------------------------- #
# local backend — run train.py on this machine (harness is on the Gaudi node)
# --------------------------------------------------------------------------- #
def run_local(artifact: str) -> dict:
    artifact = os.path.abspath(artifact)
    if not os.path.isfile(artifact):
        crash(f"artifact not found: {artifact}")
    repo_dir = os.path.dirname(artifact)  # so `import prepare` resolves

    try:
        # Inherit the container env (PT_HPU_LAZY_MODE, PT_HPU_RECIPE_CACHE_CONFIG,
        # HOME -> ~/.cache/autoresearch) so the HPU run is configured correctly.
        proc = subprocess.run(
            [sys.executable, os.path.basename(artifact)],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT,
        )
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else ""
        crash("training run exceeded timeout", log=out or "")

    output = (proc.stdout or "") + (proc.stderr or "")
    metrics = parse_metrics(output)
    if "val_bpb" not in metrics:
        crash("no val_bpb in output (run likely crashed)", log=output)
    return metrics


# --------------------------------------------------------------------------- #
# k8s backend — copy train.py into a debug pod and run it there
# --------------------------------------------------------------------------- #
def _log_tail(n=50):
    try:
        r = _run(["kubectl", "exec", "-n", NAMESPACE, POD, "--",
                  "tail", "-n", str(n), LOG_PATH], timeout=60)
        return r.stdout[-2000:]
    except Exception as e:  # noqa: BLE001
        return f"(could not read log: {e})"


def run_k8s(artifact: str) -> dict:
    # 1) Pod must be Running.
    try:
        r = _run(["kubectl", "get", "pod", POD, "-n", NAMESPACE,
                  "-o", "jsonpath={.status.phase}"], timeout=60)
    except subprocess.TimeoutExpired:
        crash("kubectl get pod timed out")
    if r.stdout.strip() != "Running":
        crash(f"debug pod not Running (phase={r.stdout.strip()!r})")

    # 2) Copy candidate train.py into the pod.
    try:
        r = _run(["kubectl", "cp", artifact, f"{NAMESPACE}/{POD}:{POD_TRAIN_PATH}"], timeout=120)
    except subprocess.TimeoutExpired:
        crash("kubectl cp timed out")
    if r.returncode != 0:
        crash(f"kubectl cp failed: {r.stderr.strip()}")

    # 3) Run the experiment (blocks ~5 min + startup).
    try:
        _run(["kubectl", "exec", "-n", NAMESPACE, POD, "--", "bash", "-c",
              f"cd /workspace/autoresearch && python train.py > {LOG_PATH} 2>&1"],
             timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        crash("training run exceeded timeout", log=_log_tail())

    # 4) Parse metrics from the log.
    try:
        r = _run(["kubectl", "exec", "-n", NAMESPACE, POD, "--",
                  "grep", "-E", "^(val_bpb|peak_vram_mb|num_steps):", LOG_PATH], timeout=60)
    except subprocess.TimeoutExpired:
        crash("grep on log timed out", log=_log_tail())

    metrics = parse_metrics(r.stdout)
    if "val_bpb" not in metrics:
        crash("no val_bpb in log (run likely crashed)", log=_log_tail())
    return metrics


def main():
    if len(sys.argv) < 2:
        crash("usage: scorer.py <path-to-train.py>")
    artifact = sys.argv[1]

    runner = detect_runner()
    metrics = run_k8s(artifact) if runner == "k8s" else run_local(artifact)

    print(json.dumps({"status": "ok", "metrics": metrics}))


if __name__ == "__main__":
    main()
