"""Rerun the multi-agent MMLU cells of Table 1 with the strict parser and pinned providers.

Same settings as the original grid (run_full_grid.build_args: 50 questions, temperature 0.2,
FinalMajorVote, pruning rate 0.25, 2 rounds). Writes to result/mmlu_rerun.csv and keeps a
full stdout log per cell in result/mmlu_rerun_logs/; never touches full_grid_results.csv.
Before each cell it checks the OpenRouter balance and stops below --min_credit.

Usage: python run_mmlu_rerun.py [--models llama qwen deepseek] [--concurrency 4] [--min_credit 15]
"""
import argparse
import asyncio
import csv
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import run_full_grid as grid

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "result" / "mmlu_rerun.csv"
LOGS = ROOT / "result" / "mmlu_rerun_logs"
METHODS = ["MAS_round1", "MAS_roundT", "AgentPrune", "AgentDropout"]
MODELS = {"llama": "meta-llama/llama-3.1-8b-instruct", "qwen": "qwen/qwen-2.5-72b-instruct",
          "deepseek": "deepseek/deepseek-chat-v3-0324"}
FIELDS = ["model", "method", "accuracy", "correct", "total", "eval_prompt_tokens", "eval_completion_tokens",
          "all_prompt_tokens", "all_completion_tokens", "cost", "failed_node_calls", "providers", "minutes", "log", "completed_at"]


def balance():
    load_dotenv(ROOT / ".env")
    req = urllib.request.Request("https://openrouter.ai/api/v1/credits",
                                 headers={"Authorization": "Bearer " + os.getenv("API_KEY", "")})
    with urllib.request.urlopen(req, timeout=60) as f:
        d = json.load(f)["data"]
    return d["total_credits"] - d["total_usage"]


def latest_usage_file(after):
    files = [p for p in (ROOT / "result" / "mmlu").glob("usage_*.json") if p.stat().st_mtime >= after]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


async def run(model, method, args, sem, lock):
    ds = next(d for d in grid.DATASETS if d.name == "MMLU")
    async with sem:
        async with lock:
            left = balance()
        if left < args.min_credit:
            print(f"SKIP {model} {method}: balance ${left:.2f} below ${args.min_credit}", flush=True)
            return
        cli = grid.build_args(ds, method, model)
        t0 = time.time()
        proc = await asyncio.create_subprocess_exec(sys.executable, "-u", f"experiments/{ds.script}", *cli, cwd=str(ROOT),
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        out, _ = await proc.communicate()
        text = out.decode("utf-8", errors="replace")
        LOGS.mkdir(parents=True, exist_ok=True)
        log = LOGS / f"{model.split('/')[-1]}_{method}_{datetime.now():%Y%m%d-%H%M%S}.log"
        log.write_text(text, encoding="utf-8")
        parsed = grid.parse_cell_output(ds, text)
        if proc.returncode != 0 or parsed is None:
            print(f"FAILED {model} {method} (exit {proc.returncode}); see {log.name}", flush=True)
            return
        correct, total, acc, ptok, ctok, cost = parsed
        usage = latest_usage_file(t0)
        ev = {"prompt_tokens": "", "completion_tokens": ""}; providers = ""
        if usage:
            u = json.loads(usage.read_text(encoding="utf-8"))
            ev = u["by_phase"].get("eval", ev)
            from collections import Counter
            providers = ";".join(f"{k}:{v}" for k, v in Counter(c.get("provider") for c in u["calls"]).items())
        row = dict(model=model, method=method, accuracy=f"{acc:.4f}", correct=correct, total=total,
                   eval_prompt_tokens=ev["prompt_tokens"], eval_completion_tokens=ev["completion_tokens"],
                   all_prompt_tokens=ptok, all_completion_tokens=ctok, cost=f"{cost:.4f}",
                   failed_node_calls=text.count("Error during execution of node"), providers=providers,
                   minutes=f"{(time.time() - t0) / 60:.1f}", log=log.name, completed_at=datetime.now().isoformat(timespec="seconds"))
        async with lock:
            new = not OUT.exists()
            with open(OUT, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
        print(f"DONE {model.split('/')[-1]:24s} {method:13s} {acc * 100:5.1f}% ({correct}/{total})  ${cost:.4f}  "
              f"failed calls {row['failed_node_calls']}  providers {providers}  {row['minutes']} min", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--min_credit", type=float, default=15.0)
    args = ap.parse_args()
    print(f"balance at start: ${balance():.2f}", flush=True)
    sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    await asyncio.gather(*(run(MODELS[m], meth, args, sem, lock) for m in args.models for meth in METHODS))
    print(f"balance at end: ${balance():.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
