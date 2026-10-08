"""
Async orchestrator for the full 3-model x 6-dataset x 6-method reproduction grid.

Each (model, dataset, method) cell is run as its own subprocess invoking the
existing experiments/run_*.py script for that dataset, with CLI args chosen to
realize the requested method. Subprocess isolation (rather than importing each
script's main() in-process) is deliberate: every run_*.py script parses its
own argparse from sys.argv, so 108 of them can't share one process's argv;
subprocesses also mean one cell crashing can't take down the whole grid or
leak state (LLM registries, torch tensors, event loops) into the next cell.

CONCURRENCY MODEL -- read this before tuning:
  There is no single global semaphore wrapping every HTTP call across the
  whole grid, because each cell is a separate OS process and Python's
  asyncio.Semaphore only works within one process. What this script actually
  controls is MAX_CONCURRENT_CELLS (how many run_*.py subprocesses are alive
  at once) and PER_CELL_BATCH_SIZE (how many questions each subprocess has
  in flight at once).

  Note: agents WITHIN one question do NOT run concurrently -- graph.arun()
  executes each round's nodes one at a time in a plain `while` loop with a
  single `await` per node (see AgentDropout/graph/graph.py, the
  zero_in_degree_queue loop), not asyncio.gather. Only the questions inside
  one batch run concurrently. So the real peak concurrent HTTP calls is
  approximately:
      MAX_CONCURRENT_CELLS * PER_CELL_BATCH_SIZE
  (independent of agent count or num_rounds). Defaults below aim for ~20
  concurrent (2 cells x batch_size 10, the batch size already validated as
  fast and reliable in every earlier manual GSM8K run). Watch for 429s in
  the logs and adjust down if seen, or up if this stays well clear of limits.

RESUMABILITY: at startup this reads result/full_grid_results.csv and skips any
(model, dataset, method) row already present, so a killed/interrupted run can
just be restarted with the same command.
"""
import asyncio
import csv
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
RESULT_CSV = ROOT / "result" / "full_grid_results.csv"
CSV_FIELDS = ["model", "dataset", "method", "accuracy", "correct_count", "total_count",
              "prompt_tokens", "completion_tokens", "total_tokens", "cost", "run_time_seconds", "completed_at"]

# ---- tunables -----------------------------------------------------------
# Both concurrency and timeout were revised after real grid data showed timeouts compounding:
# a plain 1-round Qwen-72B AQuA cell (no training) took 20 minutes on its own, and multiple
# Llama/Qwen training-method cells (AgentPrune/AgentDropout, which run 2 training phases + eval)
# timed out at the old 25min limit three times in a row -- while an unencumbered standalone run of
# one of those exact same cells was still running clean past 26 minutes. Two compounding causes:
# (1) the 72B+ models are just much slower per call than Llama-8B, and training-method cells make
# far more sequential round-trips (dec phase + opt phase + eval, each its own batch loop) than a
# single-round eval; (2) 8 concurrent cells x batch_size 5 = 40 peak concurrent HTTP calls may be
# causing silent provider-side contention (no 429s observed, but calls got slower, not just this
# process's cells -- consistent with queuing rather than hard rate-limiting). Dialing concurrency
# back to a more moderate ~25 peak and giving a much larger timeout addresses both at once; DeepSeek
# hadn't even started yet when this was tuned, so bias conservative rather than guess again.
MAX_CONCURRENT_CELLS = int(os.environ.get("GRID_MAX_CONCURRENT_CELLS", 5))
PER_CELL_BATCH_SIZE = int(os.environ.get("GRID_BATCH_SIZE", 5))
N_QUESTIONS = 50
MAX_RETRIES = 3
RETRY_BASE_DELAY = 15  # seconds; exponential: 15, 45, 135
CELL_TIMEOUT = 90 * 60  # seconds (90min) -- generous enough that even a slow training-method cell
# on the slowest model, under some contention, should finish in one attempt rather than needing luck
# across retries (a retry after a genuine-slowness timeout just repeats the same slow work).

MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "deepseek/deepseek-chat-v3-0324",
]

METHODS = ["Vanilla", "CoT", "MAS_round1", "MAS_roundT", "AgentPrune", "AgentDropout"]


@dataclass
class DatasetConfig:
    name: str
    script: str
    dataset_json: Optional[str]          # None for MMLU (uses --limit_questions instead)
    agent_name: str                      # agent used for MAS methods (>1 agent)
    mas_decision_method: str             # decision node for MAS_round1/roundT/AgentPrune/AgentDropout
    direct_agent_name: str               # agent used for Vanilla/CoT (1 agent, DirectAnswer mode)
    supports_cot: bool = True
    extra_args: list = field(default_factory=list)


DATASETS = [
    DatasetConfig("GSM8K", "run_gsm8k.py", "datasets/gsm8k/gsm8k_50.jsonl",
                  "MathSolver", "FinalMajorVote", "MathSolver"),
    DatasetConfig("AQuA", "run_aqua.py", "datasets/aqua/test_50.jsonl",
                  "MathSolver_aqua", "FinalMajorVote", "MathSolver_aqua"),
    DatasetConfig("MultiArith", "run_multiarith.py", "datasets/MultiArith/test_50.json",
                  "MathSolver", "FinalMajorVote", "MathSolver"),
    DatasetConfig("SVAMP", "run_svamp.py", "datasets/SVAMP/test_50.json",
                  "MathSolver", "FinalMajorVote", "MathSolver"),
    DatasetConfig("HumanEval", "run_humaneval.py", "datasets/humaneval/humaneval-py_50.jsonl",
                  "CodeWriting", "FinalWriteCode", "CodeWriting", supports_cot=False),
    DatasetConfig("MMLU", "run_mmlu.py", None,
                  "AnalyzeAgent", "FinalMajorVote", "AnalyzeAgent",
                  extra_args=["--limit_questions", str(N_QUESTIONS)]),
]


def build_args(ds: DatasetConfig, method: str, model: str) -> list:
    """Map (dataset, method) to the CLI args for ds.script, per the confirmed mapping:
    Vanilla = DirectAnswer/1 agent; CoT = same + --cot; MAS round=1 = FullConnected/num_rounds=1;
    MAS round=T = FullConnected/num_rounds=2; AgentPrune = +optimized_spatial/temporal/diff (no --dec);
    AgentDropout = + --dec as well."""
    args = ["--llm_name", model, "--agent_nums", "1" if method in ("Vanilla", "CoT") else "5",
            "--batch_size", str(PER_CELL_BATCH_SIZE)]
    if ds.dataset_json:
        args += ["--dataset_json", ds.dataset_json]
    args += ds.extra_args

    if method == "Vanilla":
        args += ["--mode", "DirectAnswer", "--agent_names", ds.direct_agent_name,
                  "--num_rounds", "1", "--decision_method", "FinalDirect"]
    elif method == "CoT":
        args += ["--mode", "DirectAnswer", "--agent_names", ds.direct_agent_name,
                  "--num_rounds", "1", "--decision_method", "FinalDirect", "--cot"]
    elif method == "SC_CoT":
        # Team 8: self-consistency over CoT, 5 independent samples (paper Sec. 4.1), majority vote.
        args += ["--mode", "SelfConsistency", "--agent_names", ds.direct_agent_name,
                 "--num_rounds", "1", "--decision_method", "FinalMajorVote", "--cot"]
    elif method == "MAS_round1":
        args += ["--mode", "FullConnected", "--agent_names", ds.agent_name,
                  "--num_rounds", "1", "--decision_method", ds.mas_decision_method]
    elif method == "MAS_roundT":
        args += ["--mode", "FullConnected", "--agent_names", ds.agent_name,
                  "--num_rounds", "2", "--decision_method", ds.mas_decision_method]
    elif method == "AgentPrune":
        args += ["--mode", "FullConnected", "--agent_names", ds.agent_name,
                  "--num_rounds", "2", "--decision_method", ds.mas_decision_method,
                  "--optimized_spatial", "--optimized_temporal", "--diff"]
    elif method == "AgentDropout":
        args += ["--mode", "FullConnected", "--agent_names", ds.agent_name,
                  "--num_rounds", "2", "--decision_method", ds.mas_decision_method,
                  "--optimized_spatial", "--optimized_temporal", "--diff", "--dec"]
    else:
        raise ValueError(method)
    return args


def all_cells():
    for model in MODELS:
        for ds in DATASETS:
            for method in METHODS:
                if method == "CoT" and not ds.supports_cot:
                    continue
                yield model, ds, method


# ---- output parsing -------------------------------------------------------
ACC_JSON_LOG_RE = re.compile(r'"Total solved": ([\d.]+), "Total executed": (\d+), "Accuracy": ([\d.]+)')
MMLU_ACC_RE = re.compile(r"Accuracy: [\d.]+% \((\d+)/(\d+)\)")
USAGE_RE = re.compile(r"USAGE\[TOTAL\]: prompt=(\d+) completion=(\d+) total=(\d+) calls=(\d+) cost=\$([\d.]+)")


def parse_cell_output(ds: DatasetConfig, stdout: str):
    """Returns (correct_count, total_count, accuracy, prompt_tok, completion_tok, cost) or None if unparseable."""
    prompt_tok = completion_tok = 0
    cost = 0.0
    for m in USAGE_RE.finditer(stdout):
        prompt_tok, completion_tok = int(m.group(1)), int(m.group(2))  # last match wins (re-iterated below)
        cost = float(m.group(5))
    matches = list(USAGE_RE.finditer(stdout))
    if matches:
        m = matches[-1]
        prompt_tok, completion_tok, cost = int(m.group(1)), int(m.group(2)), float(m.group(5))

    if ds.name == "MMLU":
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


# ---- CSV / resumability ----------------------------------------------------
def load_done() -> set:
    if not RESULT_CSV.exists():
        return set()
    done = set()
    with open(RESULT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row["model"], row["dataset"], row["method"]))
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


# ---- running one cell -------------------------------------------------------
async def run_one_attempt(ds: DatasetConfig, args: list):
    cmd = [sys.executable, "-u", f"experiments/{ds.script}"] + args
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(ROOT),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=CELL_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"timed out after {CELL_TIMEOUT}s")
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(stdout.splitlines()[-40:])
        raise RuntimeError(f"exit code {proc.returncode}\n--- last 40 lines ---\n{tail}")
    parsed = parse_cell_output(ds, stdout)
    if parsed is None:
        tail = "\n".join(stdout.splitlines()[-40:])
        raise RuntimeError(f"could not parse accuracy/usage from output\n--- last 40 lines ---\n{tail}")
    return parsed


async def run_cell(sem: asyncio.Semaphore, model: str, ds: DatasetConfig, method: str, state: dict):
    async with sem:
        state["in_flight"] += 1
        args = build_args(ds, method, model)
        start = time.time()
        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                correct, total, accuracy, ptok, ctok, cost = await run_one_attempt(ds, args)
                elapsed = time.time() - start
                row = {
                    "model": model, "dataset": ds.name, "method": method,
                    "accuracy": f"{accuracy:.4f}", "correct_count": correct, "total_count": total,
                    "prompt_tokens": ptok, "completion_tokens": ctok, "total_tokens": ptok + ctok,
                    "cost": f"{cost:.4f}", "run_time_seconds": f"{elapsed:.1f}",
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                }
                append_row(row)
                state["done"] += 1
                state["in_flight"] -= 1
                print(f"[{state['done']}/{state['total']} done, {state['in_flight']} in flight] "
                      f"{model} / {ds.name} / {method} -- {accuracy*100:.1f}% ({correct}/{total}), "
                      f"${cost:.4f}, {elapsed/60:.1f}m", flush=True)
                return
            except Exception as e:
                last_err = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (3 ** (attempt - 1)) * (0.8 + 0.4 * random.random())
                    print(f"[retry {attempt}/{MAX_RETRIES}] {model} / {ds.name} / {method} failed: "
                          f"{str(e).splitlines()[0]} -- retrying in {delay:.0f}s", flush=True)
                    await asyncio.sleep(delay)
        state["in_flight"] -= 1
        state["failed"] += 1
        print(f"[FAILED after {MAX_RETRIES} attempts] {model} / {ds.name} / {method}: {last_err}", flush=True)


async def main():
    smoke = "--smoke" in sys.argv
    done = load_done()
    universe = list(all_cells())
    if smoke:
        # Cheapest dataset x cheapest model x a small spread of methods, to exercise
        # concurrency/retries/resumability/CSV-writing without spending real time/money.
        wanted_methods = {"Vanilla", "MAS_round1", "AgentDropout"}
        universe = [c for c in universe if c[0] == MODELS[0] and c[1].name == "MultiArith"
                    and c[2] in wanted_methods]
        print(f"SMOKE TEST: restricting to {len(universe)} cells: "
              f"{[(m, d.name, meth) for m, d, meth in universe]}", flush=True)
    cells = [(m, d, meth) for m, d, meth in universe if (m, d.name, meth) not in done]
    total_all = len(universe)
    print(f"Grid: {total_all} total cells, {len(done)} already done, {len(cells)} to run.", flush=True)
    print(f"Concurrency: up to {MAX_CONCURRENT_CELLS} cells at once, batch_size={PER_CELL_BATCH_SIZE} per cell "
          f"(~{MAX_CONCURRENT_CELLS * PER_CELL_BATCH_SIZE} concurrent HTTP calls at peak).", flush=True)

    sem = asyncio.Semaphore(MAX_CONCURRENT_CELLS)
    state = {"done": len(done), "failed": 0, "in_flight": 0, "total": total_all}
    tasks = [asyncio.create_task(run_cell(sem, m, d, meth, state)) for m, d, meth in cells]
    await asyncio.gather(*tasks)
    print(f"Grid complete. {state['done']}/{total_all} done, {state['failed']} failed.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
