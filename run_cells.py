"""Run any set of (model, dataset, method) cells with Table 1's settings, one CSV per experiment.

Generalizes run_mmlu_rerun.py to every benchmark and adds --mode to swap the starting graph
(Table 3: Layered, Random). Arguments come from run_full_grid.build_args, so a cell here differs
from its Table 1 counterpart only in what is changed on purpose. Per cell it records accuracy,
eval-only and total tokens, cost, providers and failed node calls, and keeps the full stdout log.
Checks the OpenRouter balance before each cell and skips below --min_credit.

Usage examples:
  python run_cells.py --tag cot_rerun --models llama --datasets GSM8K AQuA MultiArith SVAMP MMLU --methods CoT
  python run_cells.py --tag table3_layered --models llama --mode Layered --methods MAS_roundT AgentPrune AgentDropout
"""
import argparse
import asyncio
import csv
import json
import os
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import run_full_grid as grid

ROOT = Path(__file__).resolve().parent
MODELS = {"llama": "meta-llama/llama-3.1-8b-instruct", "qwen": "qwen/qwen-2.5-72b-instruct",
          "deepseek": "deepseek/deepseek-chat-v3-0324"}
USAGE_DIR = {"GSM8K": "gsm8k", "AQuA": "aqua", "MultiArith": "multiarith", "SVAMP": "svamp",
             "HumanEval": "humaneval", "MMLU": "mmlu"}
FIELDS = ["model", "dataset", "method", "mode", "accuracy", "correct", "total", "eval_prompt_tokens",
          "eval_completion_tokens", "all_prompt_tokens", "all_completion_tokens", "cost", "failed_node_calls",
          "providers", "minutes", "log", "completed_at"]


def balance():
    load_dotenv(ROOT / ".env")
    req = urllib.request.Request("https://openrouter.ai/api/v1/credits",
                                 headers={"Authorization": "Bearer " + os.getenv("API_KEY", "")})
    with urllib.request.urlopen(req, timeout=60) as f:
        d = json.load(f)["data"]
    return d["total_credits"] - d["total_usage"]


def newest_usage(dataset, after):
    files = [p for p in (ROOT / "result" / USAGE_DIR[dataset]).glob("usage_*.json") if p.stat().st_mtime >= after]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


async def run(model, ds, method, a, sem, lock, out, logs, extra=(), label=""):
    """extra: flags appended to the cell's command line; label: suffix on the method name in the CSV."""
    async with sem:
        key = f"{model}|{ds.name}|{method}{label}|{a.mode or ''}"
        started = out.with_name(out.stem + "_started.txt")
        async with lock:
            if not a.redo and started.exists() and key in started.read_text(encoding="utf-8").splitlines():
                print(f"SKIP {model} {ds.name} {method}: already started under tag {a.tag} (use --redo to rerun)", flush=True)
                return
            left = balance()
            if left >= a.min_credit:
                with open(started, "a", encoding="utf-8") as f:
                    f.write(key + "\n")
        if left < a.min_credit:
            print(f"SKIP {model} {ds.name} {method}: balance ${left:.2f} below ${a.min_credit}", flush=True)
            return
        cli = grid.build_args(ds, method, model) + list(extra)
        mode = "FullConnected"
        if a.mode and "--mode" in cli and cli[cli.index("--mode") + 1] == "FullConnected":
            cli[cli.index("--mode") + 1] = a.mode
            mode = a.mode
        elif "--mode" in cli:
            mode = cli[cli.index("--mode") + 1]
        t0 = time.time()
        proc = await asyncio.create_subprocess_exec(sys.executable, "-u", f"experiments/{ds.script}", *cli, cwd=str(ROOT),
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        stdout, _ = await proc.communicate()
        text = stdout.decode("utf-8", errors="replace")
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / f"{model.split('/')[-1]}_{ds.name}_{method}{label}_{mode}_{datetime.now():%Y%m%d-%H%M%S}.log"
        log.write_text(text, encoding="utf-8")
        parsed = grid.parse_cell_output(ds, text)
        if proc.returncode != 0 or parsed is None:
            print(f"FAILED {model} {ds.name} {method}{label} {mode} (exit {proc.returncode}); see {log.name}", flush=True)
            return
        correct, total, acc, ptok, ctok, cost = parsed
        ev, providers = {"prompt_tokens": "", "completion_tokens": ""}, ""
        u = newest_usage(ds.name, t0)
        if u:
            d = json.loads(u.read_text(encoding="utf-8"))
            ev = d["by_phase"].get("eval", ev)
            providers = ";".join(f"{k}:{v}" for k, v in Counter(c.get("provider") for c in d["calls"]).items())
        row = dict(model=model, dataset=ds.name, method=method + label, mode=mode, accuracy=f"{acc:.4f}", correct=correct,
                   total=total, eval_prompt_tokens=ev["prompt_tokens"], eval_completion_tokens=ev["completion_tokens"],
                   all_prompt_tokens=ptok, all_completion_tokens=ctok, cost=f"{cost:.4f}",
                   failed_node_calls=text.count("Error during execution of node"), providers=providers,
                   minutes=f"{(time.time() - t0) / 60:.1f}", log=log.name, completed_at=datetime.now().isoformat(timespec="seconds"))
        async with lock:
            new = not out.exists()
            with open(out, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
        print(f"DONE {model.split('/')[-1]:22s} {ds.name:10s} {method + label:13s} {mode:13s} {acc * 100:5.1f}% ({correct}/{total})  "
              f"${cost:.4f}  failed {row['failed_node_calls']}  {providers}  {row['minutes']} min", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="experiment name: results go to result/<tag>.csv")
    ap.add_argument("--models", nargs="+", default=["llama"], choices=list(MODELS))
    ap.add_argument("--datasets", nargs="+", default=[d.name for d in grid.DATASETS], choices=[d.name for d in grid.DATASETS])
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--mode", default=None, help="replace the FullConnected starting graph (e.g. Layered, Random)")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--min_credit", type=float, default=5.0)
    ap.add_argument("--redo", action="store_true", help="rerun cells already started under this tag")
    a = ap.parse_args()
    out, logs = ROOT / "result" / f"{a.tag}.csv", ROOT / "result" / f"{a.tag}_logs"
    print(f"{a.tag}: balance at start ${balance():.2f}", flush=True)
    sem, lock = asyncio.Semaphore(a.concurrency), asyncio.Lock()
    cells = [(MODELS[m], d, meth) for m in a.models for d in grid.DATASETS if d.name in a.datasets for meth in a.methods
             if not (meth in ("CoT", "SC_CoT") and not d.supports_cot)]
    await asyncio.gather(*(run(m, d, meth, a, sem, lock, out, logs) for m, d, meth in cells))
    print(f"{a.tag}: balance at end ${balance():.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
