"""
Pruning-rate sweep: AgentDropout at 5 pruning_rate values x 2 models x 2 datasets
(GSM8K, MMLU), 50 questions each. Mirrors the paper's Section 4.3 ablation.

Same subprocess-per-cell design as run_full_grid.py, deliberately kept separate
(own CSV, own script) so this doesn't touch the main grid's resumable results file.
"""
import asyncio
import csv
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULT_CSV = ROOT / "result" / "pruning_rate_sweep.csv"
CSV_FIELDS = ["model", "dataset", "pruning_rate", "accuracy", "correct_count", "total_count",
              "prompt_tokens", "completion_tokens", "total_tokens", "cost", "run_time_seconds", "completed_at"]

MAX_CONCURRENT_CELLS = 4
PER_CELL_BATCH_SIZE = 5
CELL_TIMEOUT = 90 * 60
MAX_RETRIES = 3
RETRY_BASE_DELAY = 15

MODELS = ["meta-llama/llama-3.1-8b-instruct", "qwen/qwen-2.5-72b-instruct"]
RATES = [0.05, 0.10, 0.15, 0.20, 0.30]

ACC_JSON_LOG_RE = re.compile(r'"Total solved": ([\d.]+), "Total executed": (\d+), "Accuracy": ([\d.]+)')
MMLU_ACC_RE = re.compile(r"Accuracy: [\d.]+% \((\d+)/(\d+)\)")
USAGE_RE = re.compile(r"USAGE\[TOTAL\]: prompt=(\d+) completion=(\d+) total=(\d+) calls=(\d+) cost=\$([\d.]+)")


def build_args(dataset: str, model: str, rate: float) -> list:
    # --agent_nums 5 matters: the node-dropout logic in graph.py hardcodes indexing
    # for exactly 5 agents (the paper's own setup) and crashes on any other count.
    common = ["--llm_name", model, "--agent_nums", "5", "--batch_size", str(PER_CELL_BATCH_SIZE),
              "--mode", "FullConnected", "--num_rounds", "2",
              "--decision_method", "FinalMajorVote",
              "--optimized_spatial", "--optimized_temporal", "--diff", "--dec",
              "--pruning_rate", str(rate)]
    if dataset == "GSM8K":
        return ["run_gsm8k.py"] + common + ["--dataset_json", "datasets/gsm8k/gsm8k_50.jsonl",
                                             "--agent_names", "MathSolver"]
    elif dataset == "MMLU":
        return ["run_mmlu.py"] + common + ["--agent_names", "AnalyzeAgent", "--limit_questions", "50"]
    raise ValueError(dataset)


def parse_cell_output(dataset: str, stdout: str):
    prompt_tok = completion_tok = 0
    cost = 0.0
    matches = list(USAGE_RE.finditer(stdout))
    if matches:
        m = matches[-1]
        prompt_tok, completion_tok, cost = int(m.group(1)), int(m.group(2)), float(m.group(5))
    if dataset == "MMLU":
        acc_matches = list(MMLU_ACC_RE.finditer(stdout))
        if not acc_matches:
            return None
        correct, total = int(acc_matches[-1].group(1)), int(acc_matches[-1].group(2))
    else:
        log_matches = list(ACC_JSON_LOG_RE.finditer(stdout))
        if not log_matches:
            return None
        correct, total = float(log_matches[-1].group(1)), int(log_matches[-1].group(2))
    accuracy = correct / total if total else 0.0
    return correct, total, accuracy, prompt_tok, completion_tok, cost


def load_done() -> set:
    if not RESULT_CSV.exists():
        return set()
    done = set()
    with open(RESULT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row["model"], row["dataset"], row["pruning_rate"]))
    return done


def append_row(row: dict):
    new_file = not RESULT_CSV.exists()
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)
        f.flush()


async def run_one_attempt(dataset: str, args: list):
    cmd = [sys.executable, "-u", f"experiments/{args[0]}"] + args[1:]
    proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(ROOT),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=CELL_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill(); await proc.wait()
        raise RuntimeError(f"timed out after {CELL_TIMEOUT}s")
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(stdout.splitlines()[-40:])
        raise RuntimeError(f"exit code {proc.returncode}\n{tail}")
    parsed = parse_cell_output(dataset, stdout)
    if parsed is None:
        tail = "\n".join(stdout.splitlines()[-40:])
        raise RuntimeError(f"could not parse output\n{tail}")
    return parsed


async def run_cell(sem, model, dataset, rate, state):
    async with sem:
        args = build_args(dataset, model, rate)
        start = time.time()
        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                correct, total, accuracy, ptok, ctok, cost = await run_one_attempt(dataset, args)
                elapsed = time.time() - start
                row = {"model": model, "dataset": dataset, "pruning_rate": rate,
                       "accuracy": f"{accuracy:.4f}", "correct_count": correct, "total_count": total,
                       "prompt_tokens": ptok, "completion_tokens": ctok, "total_tokens": ptok + ctok,
                       "cost": f"{cost:.4f}", "run_time_seconds": f"{elapsed:.1f}",
                       "completed_at": datetime.now().isoformat(timespec="seconds")}
                append_row(row)
                state["done"] += 1
                print(f"[{state['done']}/{state['total']}] {model} / {dataset} / rate={rate} -- "
                      f"{accuracy*100:.1f}% ({correct}/{total}), ${cost:.4f}, {elapsed/60:.1f}m", flush=True)
                return
            except Exception as e:
                last_err = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (3 ** (attempt - 1))
                    print(f"[retry {attempt}/{MAX_RETRIES}] {model}/{dataset}/rate={rate} failed: "
                          f"{str(e).splitlines()[0]} -- retrying in {delay:.0f}s", flush=True)
                    await asyncio.sleep(delay)
        state["failed"] += 1
        print(f"[FAILED] {model}/{dataset}/rate={rate}: {last_err}", flush=True)


async def main():
    done = load_done()
    cells = [(m, d, r) for m in MODELS for d in ("GSM8K", "MMLU") for r in RATES
             if (m, d, str(r)) not in done]
    total_all = len(MODELS) * 2 * len(RATES)
    print(f"Sweep: {total_all} total cells, {len(done)} already done, {len(cells)} to run.", flush=True)
    sem = asyncio.Semaphore(MAX_CONCURRENT_CELLS)
    state = {"done": len(done), "failed": 0, "total": total_all}
    tasks = [asyncio.create_task(run_cell(sem, m, d, r, state)) for m, d, r in cells]
    await asyncio.gather(*tasks)
    print(f"Sweep complete. {state['done']}/{total_all} done, {state['failed']} failed.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
