"""Table 6 (domain transferability): learn the AgentDropout graph on benchmark A, evaluate it on benchmark B.

For each A in GSM8K, AQuA, MultiArith, SVAMP: one full AgentDropout run on A with --save_graph (its own
accuracy is the diagonal cell A -> A), then, as soon as that graph exists, three eval-only runs on the other
benchmarks with --load_graph (training skipped). All other settings are Table 1's, via run_cells/build_args.
Rows land in result/table6.csv with method "AgentDropout_from_<A>". Safe to rerun: started cells are skipped.

Usage: python run_table6.py [--concurrency 2] [--min_credit 15] [--sources GSM8K AQuA MultiArith SVAMP]
"""
import argparse
import asyncio
from pathlib import Path

import run_cells as rc
import run_full_grid as grid

MATH = ["GSM8K", "AQuA", "MultiArith", "SVAMP"]


async def source(A, a, sem, lock, out, logs, graphs):
    ds = {d.name: d for d in grid.DATASETS}
    model = rc.MODELS["llama"]
    path = graphs / f"llama_{A}.pt"
    label = f"_from_{A}"
    if not path.exists():
        await rc.run(model, ds[A], "AgentDropout", a, sem, lock, out, logs, extra=["--save_graph", str(path)], label=label)
    if not path.exists():
        print(f"SKIP transfers from {A}: no saved graph at {path.name} (training cell failed or was skipped)", flush=True)
        return
    await asyncio.gather(*(rc.run(model, ds[B], "AgentDropout", a, sem, lock, out, logs,
                                  extra=["--load_graph", str(path)], label=label) for B in MATH if B != A))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=MATH, choices=MATH)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--min_credit", type=float, default=15.0)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    a.tag, a.mode = "table6", None
    out, logs = rc.ROOT / "result" / "table6.csv", rc.ROOT / "result" / "table6_logs"
    graphs = rc.ROOT / "result" / "table6_graphs"
    graphs.mkdir(parents=True, exist_ok=True)
    print(f"table6: balance at start ${rc.balance():.2f}", flush=True)
    sem, lock = asyncio.Semaphore(a.concurrency), asyncio.Lock()
    await asyncio.gather(*(source(A, a, sem, lock, out, logs, graphs) for A in a.sources))
    print(f"table6: balance at end ${rc.balance():.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
