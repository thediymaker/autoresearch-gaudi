#!/usr/bin/env python3
"""Model-agnostic autotuning harness.

Drives a LOCAL OpenAI-compatible LLM (the "brain") through an experiment loop:
the model proposes an edit to a single artifact, the harness runs a fixed scorer
to measure it, and the model decides keep/discard. The brain only ever sees a
small set of bounded tools (no arbitrary shell), so it cannot escape the loop.

The experiment is fully pluggable via a directory containing:
  experiment.json  - config (artifact, scorer, metric, goal, program)
  program.md       - natural-language rules shown to the model
  scorer.py        - fixed code the harness runs to measure the artifact
                     (NOT exposed to the model). Must print a JSON object to
                     stdout: {"status": "ok"|"crash", "metrics": {<metric>: float, ...}}

Config (env):
  OPENAI_BASE_URL   OpenAI-compatible API base, e.g. http://localhost:8000/v1
  OPENAI_API_KEY    bearer key for that endpoint
  OPENAI_MODEL      model name to request, e.g. my-model

Usage:
  python run_loop.py --experiment ../experiments/gpt_train --iterations 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1").rstrip("/")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("OPENAI_MODEL", "my-model")

REQUEST_TIMEOUT = 300          # seconds per chat completion
MAX_TOOL_STEPS = 40            # tool calls allowed per iteration before forcing a decision
SCORER_TIMEOUT = 900           # seconds per experiment run (10 min budget + headroom)
IMPROVE_MARGIN_FRAC = 0.002    # a keep must beat best by >=0.2% to clear run-to-run noise
WARM_BASELINE = True           # run baseline twice (warm the HPU recipe cache), use 2nd


# --------------------------------------------------------------------------- #
# OpenAI-compatible chat client (stdlib only)
# --------------------------------------------------------------------------- #
def chat(messages, tools, retries=4):
    """One /v1/chat/completions call. Returns the assistant message dict.

    Retries transient gateway failures (5xx, timeouts, connection resets) with
    exponential backoff so one bad response doesn't kill the whole run.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 8000,
    }
    body = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(retries):
        req = urllib.request.Request(
            f"{BASE_URL}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]
        except urllib.error.HTTPError as e:
            last_err = e
            # 4xx (except 429) are non-transient; don't retry.
            if e.code < 500 and e.code != 429:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
        wait = 2 ** attempt
        print(f"[chat] transient error ({last_err}); retry {attempt + 1}/{retries} in {wait}s", flush=True)
        time.sleep(wait)
    raise last_err


# --------------------------------------------------------------------------- #
# Tool schema exposed to the model (bounded — no arbitrary shell)
# --------------------------------------------------------------------------- #
def build_tools(metric):
    return [
        {
            "type": "function",
            "function": {
                "name": "read_artifact",
                "description": "Return the full current contents of the artifact file you are editing.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_artifact",
                "description": (
                    "Replace exactly one occurrence of old_str with new_str in the artifact. "
                    "old_str must match the file exactly (including whitespace) and be unique."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "old_str": {"type": "string", "description": "Exact text to replace (must be unique)."},
                        "new_str": {"type": "string", "description": "Replacement text."},
                    },
                    "required": ["old_str", "new_str"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_experiment",
                "description": (
                    "Run the fixed scorer on the current artifact and return measured metrics "
                    f"(including '{metric}') plus status. Takes several minutes."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "keep",
                "description": (
                    "Accept the current artifact as the new best and end this iteration. "
                    "Only call after run_experiment shows an improvement you want to keep."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string", "description": "Short summary of what this change did."},
                    },
                    "required": ["description"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "discard",
                "description": (
                    "Reject the current artifact, revert to the previous best, and end this iteration."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "Why this change is being discarded."},
                    },
                    "required": ["reason"],
                },
            },
        },
    ]


# --------------------------------------------------------------------------- #
# Tool implementations (operate on real files; harness owns all side effects)
# --------------------------------------------------------------------------- #
class Experiment:
    def __init__(self, exp_dir: Path):
        self.dir = exp_dir.resolve()
        cfg = json.loads((self.dir / "experiment.json").read_text())
        self.name = cfg["name"]
        self.metric = cfg["metric"]
        self.goal = cfg["goal"]  # "minimize" or "maximize"
        self.artifact = (self.dir / cfg["artifact"]).resolve()
        self.scorer = (self.dir / cfg["scorer"]).resolve()
        self.program = (self.dir / cfg.get("program", "program.md")).read_text()

        # Harness-owned state files (kept inside the experiment dir, gitignored).
        state = self.dir / ".harness"
        state.mkdir(exist_ok=True)
        self.best_backup = state / "best_artifact"
        self.results_tsv = self.dir / "results.tsv"

        if not self.best_backup.exists():
            shutil.copy2(self.artifact, self.best_backup)
        if not self.results_tsv.exists():
            self.results_tsv.write_text("hash\t%s\tmemory_gb\tstatus\tdescription\n" % self.metric)

        self.best_metric = None  # set after the baseline run
        self.best_sha = self._sha(self.best_backup)  # sha of the current best artifact

    @staticmethod
    def _sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]

    def artifact_sha(self):
        return self._sha(self.artifact)

    def is_better(self, candidate, best):
        if best is None:
            return True
        return candidate < best if self.goal == "minimize" else candidate > best

    def beats_margin(self, candidate, best):
        """True only if candidate beats best by at least IMPROVE_MARGIN_FRAC."""
        if best is None:
            return True
        margin = abs(best) * IMPROVE_MARGIN_FRAC
        if self.goal == "minimize":
            return candidate <= best - margin
        return candidate >= best + margin

    # -- tool: read_artifact ------------------------------------------------- #
    def read_artifact(self):
        return self.artifact.read_text()

    # -- tool: edit_artifact ------------------------------------------------- #
    def edit_artifact(self, old_str, new_str):
        text = self.artifact.read_text()
        count = text.count(old_str)
        if count == 0:
            return {"ok": False, "error": "old_str not found in artifact."}
        if count > 1:
            return {"ok": False, "error": f"old_str matches {count} times; make it unique."}
        self.artifact.write_text(text.replace(old_str, new_str, 1))
        return {"ok": True, "message": "Edit applied."}

    # -- tool: run_experiment ------------------------------------------------ #
    def run_experiment(self, allow_unchanged=False):
        # Guardrail: refuse to score an artifact identical to the current best.
        # This blocks the "narrate a change in text but never edit, then keep the
        # warmup noise" failure mode. The baseline/warmup runs pass allow_unchanged.
        if not allow_unchanged and self.artifact_sha() == self.best_sha:
            return {
                "status": "noop",
                "metrics": {},
                "error": (
                    "Artifact is UNCHANGED from the current best (no edit detected). "
                    "Call edit_artifact with a real old_str/new_str change before running. "
                    "Describing a change in text does nothing — you must use the edit tool."
                ),
            }
        try:
            proc = subprocess.run(
                [sys.executable, str(self.scorer), str(self.artifact)],
                capture_output=True,
                text=True,
                timeout=SCORER_TIMEOUT,
                cwd=str(self.dir),
            )
        except subprocess.TimeoutExpired:
            return {"status": "crash", "metrics": {}, "error": "scorer timed out"}
        out = proc.stdout.strip()
        # Scorer must print a JSON object as its last stdout line.
        result = None
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    result = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        if result is None:
            return {
                "status": "crash",
                "metrics": {},
                "error": "scorer produced no JSON result",
                "scorer_stderr_tail": proc.stderr.strip()[-1500:],
            }
        return result

    # -- tool: keep ---------------------------------------------------------- #
    def keep(self, description, last_result):
        # Guardrail 1: cannot keep an unedited artifact.
        if self.artifact_sha() == self.best_sha:
            return {"ok": False, "error": "Cannot keep: artifact is unchanged from the current best. Edit it first."}
        # Guardrail 2: must have a successful measurement of THIS artifact.
        if not last_result or last_result.get("status") != "ok":
            return {"ok": False, "error": "Cannot keep: no successful run_experiment result for the current artifact."}
        metric_val = last_result.get("metrics", {}).get(self.metric)
        if not isinstance(metric_val, (int, float)):
            return {"ok": False, "error": f"Cannot keep: last run produced no '{self.metric}' metric."}
        # Guardrail 3: must beat the best by a real margin (not run-to-run noise).
        if not self.beats_margin(metric_val, self.best_metric):
            return {
                "ok": False,
                "error": (
                    f"Cannot keep: {self.metric}={metric_val} does not beat best={self.best_metric} "
                    f"by the required {IMPROVE_MARGIN_FRAC:.1%} margin. Call discard instead."
                ),
            }
        self._record(last_result, "keep", description)
        self.best_metric = metric_val
        shutil.copy2(self.artifact, self.best_backup)
        self.best_sha = self.artifact_sha()
        return {"ok": True, "best_metric": self.best_metric}

    # -- tool: discard ------------------------------------------------------- #
    def discard(self, reason, last_result):
        status = "crash" if (last_result or {}).get("status") == "crash" else "discard"
        self._record(last_result, status, reason)
        shutil.copy2(self.best_backup, self.artifact)
        return {"ok": True, "reverted_to_best_metric": self.best_metric}

    # -- results.tsv --------------------------------------------------------- #
    def _record(self, last_result, status, description):
        metrics = (last_result or {}).get("metrics", {})
        metric_val = metrics.get(self.metric)
        peak_mb = metrics.get("peak_vram_mb")
        h = hashlib.sha256(self.artifact.read_bytes()).hexdigest()[:7]
        mv = f"{metric_val:.6f}" if isinstance(metric_val, (int, float)) else "0.000000"
        gb = f"{peak_mb / 1024:.1f}" if isinstance(peak_mb, (int, float)) else "0.0"
        desc = description.replace("\t", " ").replace("\n", " ").strip()
        with self.results_tsv.open("a") as f:
            f.write(f"{h}\t{mv}\t{gb}\t{status}\t{desc}\n")

    # -- restart support ----------------------------------------------------- #
    def seed_best_from_results(self):
        """Recover best_metric from the last kept row in results.tsv (for restarts)."""
        best = None
        for line in self.results_tsv.read_text().splitlines()[1:]:
            cols = line.split("\t")
            if len(cols) < 4 or cols[3] not in ("keep",):
                continue
            try:
                val = float(cols[1])
            except ValueError:
                continue
            if self.is_better(val, best):
                best = val
        self.best_metric = best
        self.best_sha = self._sha(self.best_backup)
        return best


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are an autonomous optimization agent operating inside a bounded harness.

You may ONLY act through the provided tools. You cannot run shell commands; the
harness runs the fixed scorer for you when you call run_experiment.

CRITICAL RULES (the harness enforces these — violating them wastes the iteration):
- To change the artifact you MUST call edit_artifact with an exact old_str/new_str.
  Writing what you "would" change in plain text does NOTHING. Always read_artifact
  first so your old_str matches the file exactly.
- run_experiment is REJECTED if the artifact is unchanged from the current best.
  You must make a real edit before running.
- keep is REJECTED unless (a) the artifact was actually edited, (b) you have a
  successful run_experiment result for it, and (c) it beats the current best by a
  real margin. If your result does not clearly beat the best, call discard.

Each iteration: make ONE coherent experimental change, run it, read the measured
metric, then decide. You MUST end every iteration with exactly one keep or discard.

Workflow per iteration:
1. read_artifact to see the current code.
2. edit_artifact (one or more real edits) to apply your idea.
3. run_experiment to measure it.
4. keep(description) ONLY if the metric clearly beat the best; otherwise discard(reason).

Be decisive and honest. Do not fabricate results. Do not ask the human anything.
Follow the experiment rules below."""


def run_baseline(exp: Experiment):
    """Establish the baseline metric before the model starts iterating.

    On Gaudi the training script has a FIXED wall-clock budget, so a cold HPU
    recipe cache (graphs compiling during the budget) inflates the metric. We run
    once to warm the cache, then a second time for the real, fair baseline.
    """
    if WARM_BASELINE:
        print(f"[harness] warming HPU recipe cache for '{exp.name}' (throwaway run) ...", flush=True)
        t_w = time.time()
        warm = exp.run_experiment(allow_unchanged=True)
        wv = warm.get("metrics", {}).get(exp.metric)
        print(f"[harness] warmup {exp.metric} = {wv} (discarded, {time.time() - t_w:.0f}s)", flush=True)
    print(f"[harness] running baseline for '{exp.name}' ...", flush=True)
    t_b = time.time()
    result = exp.run_experiment(allow_unchanged=True)
    metric_val = result.get("metrics", {}).get(exp.metric)
    if result.get("status") != "ok" or metric_val is None:
        print(f"[harness] BASELINE FAILED: {result}", file=sys.stderr, flush=True)
        sys.exit(1)
    exp.best_metric = metric_val
    exp._record(result, "keep", "baseline")
    shutil.copy2(exp.artifact, exp.best_backup)
    exp.best_sha = exp.artifact_sha()
    print(f"[harness] baseline {exp.metric} = {metric_val} (goal: {exp.goal}, {time.time() - t_b:.0f}s)", flush=True)


def _short(text, limit=200):
    """Collapse whitespace and truncate for one-line console logging."""
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[: limit - 1] + "\u2026"


def _describe_call(name, args):
    """Human-readable summary of a tool call so you can watch what the model does."""
    if name == "edit_artifact":
        return f"replace [{_short(args.get('old_str', ''), 70)}] -> [{_short(args.get('new_str', ''), 70)}]"
    if name == "keep":
        return f"KEEP \u2014 {_short(args.get('description', ''), 160)}"
    if name == "discard":
        return f"DISCARD \u2014 {_short(args.get('reason', ''), 160)}"
    if name == "run_experiment":
        return "score the current artifact (training run \u2014 several minutes)\u2026"
    if name == "read_artifact":
        return "re-read the artifact"
    return "(" + ", ".join(args.keys()) + ")"


def iteration(exp: Experiment, history: str, idx: int) -> str:
    """Run one model-driven iteration. Returns a one-line history summary."""
    tools = build_tools(exp.metric)
    artifact_text = exp.read_artifact()
    user = (
        f"{exp.program}\n\n"
        f"=== CURRENT STATE ===\n"
        f"Metric: {exp.metric} (goal: {exp.goal})\n"
        f"Current best {exp.metric}: {exp.best_metric}\n"
        f"Iteration: {idx}\n\n"
        f"=== HISTORY SO FAR ===\n{history or '(none yet)'}\n\n"
        f"=== CURRENT ARTIFACT ({exp.artifact.name}) ===\n"
        f"The full current file is below. You already have it — do NOT call read_artifact "
        f"unless you truly need to re-check it. Use edit_artifact with exact substrings from it.\n"
        f"```\n{artifact_text}\n```\n\n"
        f"Begin iteration {idx}. Make one real edit_artifact change, run_experiment, then keep or discard."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

    last_result = None
    reads = 0
    for step in range(MAX_TOOL_STEPS):
        print(f"[iter {idx} step {step}] brain thinking ...", flush=True)
        t_brain = time.time()
        msg = chat(messages, tools)
        dt = time.time() - t_brain
        messages.append(msg)
        tool_calls = msg.get("tool_calls") or []

        # Surface the model's own words (and any reasoning trace) so you can
        # follow its thinking and tell when it is just stalling on the endpoint.
        reasoning = (msg.get("content") or msg.get("reasoning_content")
                     or msg.get("reasoning") or "").strip()
        if reasoning:
            print(f"[iter {idx} step {step}] brain ({dt:.0f}s): {_short(reasoning, 500)}", flush=True)
        else:
            print(f"[iter {idx} step {step}] brain replied in {dt:.0f}s", flush=True)

        if not tool_calls:
            # Model spoke without acting; nudge it to use a tool.
            messages.append({
                "role": "user",
                "content": "You must act through tools and end with keep or discard.",
            })
            continue

        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            print(f"[iter {idx} step {step}] -> {name}: {_describe_call(name, args)}", flush=True)

            if name == "read_artifact":
                reads += 1
                # The artifact is already in the opening prompt; cap re-reads to
                # avoid ballooning context (which trips gateway 5xx limits).
                if reads > 1:
                    result = {
                        "note": "Artifact already provided in the opening message. "
                                "Stop re-reading and make an edit_artifact change now."
                    }
                else:
                    result = exp.read_artifact()
            elif name == "edit_artifact":
                result = exp.edit_artifact(args.get("old_str", ""), args.get("new_str", ""))
            elif name == "run_experiment":
                t_run = time.time()
                result = exp.run_experiment()
                last_result = result
                mv = result.get("metrics", {}).get(exp.metric)
                print(f"[iter {idx}] experiment {exp.metric}={mv} status={result.get('status')} "
                      f"({time.time() - t_run:.0f}s)", flush=True)
            elif name == "keep":
                r = exp.keep(args.get("description", ""), last_result)
                if r.get("ok"):
                    return f"iter {idx}: KEEP — {args.get('description','')} (best {exp.metric}={r['best_metric']})"
                # Keep was rejected by a guardrail; tell the model and continue.
                print(f"[iter {idx}] keep REJECTED: {r.get('error')}", flush=True)
                result = r
            elif name == "discard":
                exp.discard(args.get("reason", ""), last_result)
                return f"iter {idx}: DISCARD — {args.get('reason','')}"
            else:
                result = {"error": f"unknown tool {name}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result if isinstance(result, str) else json.dumps(result),
            })

    # Ran out of tool steps without a decision — force a discard to stay safe.
    exp.discard("iteration exceeded max tool steps without a decision", last_result)
    return f"iter {idx}: DISCARD (forced — no decision within step budget)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="path to experiment directory")
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--skip-baseline", action="store_true",
                    help="assume best_metric from prior run instead of running baseline")
    args = ap.parse_args()

    if not API_KEY:
        print("[harness] OPENAI_API_KEY is not set", file=sys.stderr)
        sys.exit(2)

    exp = Experiment(Path(args.experiment))
    print(f"[harness] model={MODEL} base={BASE_URL} experiment={exp.name}", flush=True)

    if args.skip_baseline:
        seeded = exp.seed_best_from_results()
        if seeded is None:
            print("[harness] --skip-baseline but no kept result found; running baseline", flush=True)
            run_baseline(exp)
        else:
            # Ensure the artifact matches the recorded best before iterating.
            shutil.copy2(exp.best_backup, exp.artifact)
            print(f"[harness] resuming: best {exp.metric} = {seeded} (from results.tsv)", flush=True)
    else:
        # If a previous run was killed mid-iteration (before keep/discard), the
        # artifact on disk may be a broken, un-reverted edit. Restore the known-
        # good backup so the baseline always starts from a clean, working state.
        if exp.best_backup.exists():
            shutil.copy2(exp.best_backup, exp.artifact)
            print("[harness] restored artifact from last known-good backup", flush=True)
        run_baseline(exp)

    history_lines = []
    for idx in range(1, args.iterations + 1):
        t0 = time.time()
        summary = iteration(exp, "\n".join(history_lines), idx)
        history_lines.append(summary)
        print(f"[harness] {summary}  ({time.time() - t0:.0f}s)", flush=True)

    print(f"[harness] done. best {exp.metric} = {exp.best_metric}", flush=True)
    print(f"[harness] results: {exp.results_tsv}", flush=True)


if __name__ == "__main__":
    main()
