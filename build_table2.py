"""Assemble the reproduction of the paper's Table 2 (Llama token consumption, inference only) from saved results.

Sources (50 evaluation questions per cell; HumanEval excluded by team decision, 8 Oct 2026):
  Vanilla, CoT, MAS (r=1), MAS (r=T): full_grid_results.csv. These methods do no training, so the whole-run
      totals are inference tokens. Llama CoT on the four math sets comes from cot_rerun.csv (prompt fix).
  MMLU multi-agent rows: mmlu_rerun.csv (corrected parser and answer-format line), eval phase only.
  AgentPrune, AgentDropout, SC (CoT) elsewhere: table2.csv, eval phase only (training calls excluded,
      as in the paper).
The paper evaluates full test sets, so absolute counts are compared per question and through the relative
reduction of AgentDropout against MAS (r=T) and AgentPrune.
Usage: python build_table2.py   (writes result/table2_assembled.csv)
"""
import csv
from pathlib import Path

R = Path(__file__).resolve().parent / "result"
LLAMA = "meta-llama/llama-3.1-8b-instruct"
DATASETS = ["MMLU", "GSM8K", "AQuA", "MultiArith", "SVAMP"]
METHODS = ["Vanilla", "CoT", "SC_CoT", "MAS_round1", "MAS_roundT", "AgentPrune", "AgentDropout"]
LABEL = {"SC_CoT": "SC (CoT)", "MAS_round1": "MAS (r=1)", "MAS_roundT": "MAS (r=T)"}
PAPER_N = {"MMLU": 153, "GSM8K": 1319, "AQuA": 254, "MultiArith": 600, "SVAMP": 1000}
K, M = 1e3, 1e6
PAPER = {  # paper Table 2 (Ptok, Ctok), full test sets
    "Vanilla": [(99 * K, 44 * K), (142 * K, 209 * K), (32 * K, 89 * K), (16 * K, 15 * K), (27 * K, 19 * K)],
    "CoT": [(129 * K, 70 * K), (154 * K, 337 * K), (34 * K, 105 * K), (18 * K, 23 * K), (30 * K, 50 * K)],
    "SC_CoT": [(645 * K, 346 * K), (770 * K, 1.7 * M), (170 * K, 594 * K), (90 * K, 102 * K), (150 * K, 281 * K)],
    "MAS_round1": [(1.4 * M, 355 * K), (8.5 * M, 1.9 * M), (1.1 * M, 390 * K), (1.1 * M, 218 * K), (1.9 * M, 402 * K)],
    "MAS_roundT": [(1.6 * M, 387 * K), (16 * M, 3.4 * M), (2.4 * M, 745 * K), (2.1 * M, 388 * K), (3.7 * M, 721 * K)],
    "AgentPrune": [(1.3 * M, 367 * K), (15 * M, 3.6 * M), (2.0 * M, 759 * K), (1.9 * M, 393 * K), (3.4 * M, 714 * K)],
    "AgentDropout": [(1.1 * M, 333 * K), (12 * M, 2.8 * M), (1.3 * M, 634 * K), (1.4 * M, 312 * K), (2.6 * M, 594 * K)],
}


def rows(name):
    p = R / name
    return list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []


ours, acc, src = {}, {}, {}
for r in rows("full_grid_results.csv"):
    if r["model"] == LLAMA and r["dataset"] in DATASETS and r["method"] in ("Vanilla", "CoT", "MAS_round1", "MAS_roundT"):
        ours[(r["dataset"], r["method"])] = (float(r["prompt_tokens"]), float(r["completion_tokens"]))
        acc[(r["dataset"], r["method"])] = float(r["accuracy"])
        src[(r["dataset"], r["method"])] = "grid"
for r in rows("cot_rerun.csv"):
    if r["model"] == LLAMA:
        ours[(r["dataset"], "CoT")] = (float(r["eval_prompt_tokens"]), float(r["eval_completion_tokens"]))
        acc[(r["dataset"], "CoT")] = float(r["accuracy"])
        src[(r["dataset"], "CoT")] = "cot_rerun"
for r in rows("mmlu_rerun.csv"):
    if r["model"] == LLAMA:
        ours[("MMLU", r["method"])] = (float(r["eval_prompt_tokens"]), float(r["eval_completion_tokens"]))
        acc[("MMLU", r["method"])] = float(r["accuracy"])
        src[("MMLU", r["method"])] = "mmlu_rerun"
# MMLU single-agent accuracy re-scored offline with the strict parser (tools/verification/mmlu_revote.py);
# the grid CSV holds the old first-character score. Tokens are unaffected.
acc[("MMLU", "Vanilla")], acc[("MMLU", "CoT")] = 0.52, 0.52
for r in rows("table2.csv"):
    if r["model"] == LLAMA and r["dataset"] in DATASETS:
        ours[(r["dataset"], r["method"])] = (float(r["eval_prompt_tokens"]), float(r["eval_completion_tokens"]))
        acc[(r["dataset"], r["method"])] = float(r["accuracy"])
        src[(r["dataset"], r["method"])] = "table2"


def fmt(x):
    return f"{x / M:.2f}M" if x >= M else f"{x / K:.0f}K"


missing = [(d, m) for d in DATASETS for m in METHODS if (d, m) not in ours]
print("missing cells:", missing or "none")
print(f"\nOURS (50 questions per cell): Ptok / Ctok, accuracy in parentheses")
print(f"{'Method':13s} " + " ".join(f"{d:>22s}" for d in DATASETS))
for m in METHODS:
    cells = []
    for d in DATASETS:
        if (d, m) in ours:
            p, c = ours[(d, m)]
            cells.append(f"{fmt(p)}/{fmt(c)} ({acc[(d, m)] * 100:.0f}%)")
        else:
            cells.append("-")
    print(f"{LABEL.get(m, m):13s} " + " ".join(f"{x:>22s}" for x in cells))


def reduction(table, d, a, b, k):
    return 100 * (1 - table[(d, a)][k] / table[(d, b)][k]) if (d, a) in table and (d, b) in table else None


paper = {(d, m): PAPER[m][i] for m in PAPER for i, d in enumerate(DATASETS)}
print("\nAgentDropout token reduction (prompt / completion), ours vs paper")
out = []
for ref in ["MAS_roundT", "AgentPrune"]:
    print(f"  vs {LABEL.get(ref, ref)}:")
    for d in DATASETS:
        o = [reduction(ours, d, "AgentDropout", ref, k) for k in (0, 1)]
        p = [reduction(paper, d, "AgentDropout", ref, k) for k in (0, 1)]
        s = lambda v: "  n/a" if v is None else f"{v:+5.1f}%"
        print(f"    {d:10s} ours {s(o[0])} / {s(o[1])}   paper {s(p[0])} / {s(p[1])}")
        out.append(dict(reference=ref, dataset=d, ours_prompt=o[0], ours_completion=o[1], paper_prompt=p[0],
                        paper_completion=p[1]))
    have = [d for d in DATASETS if (d, "AgentDropout") in ours and (d, ref) in ours]
    if have:
        for name, t in (("ours", ours), ("paper", paper)):
            pr = 100 * (1 - sum(t[(d, "AgentDropout")][0] for d in have) / sum(t[(d, ref)][0] for d in have))
            cr = 100 * (1 - sum(t[(d, "AgentDropout")][1] for d in have) / sum(t[(d, ref)][1] for d in have))
            print(f"    {'pooled ' + name:16s} {pr:+5.1f}% / {cr:+5.1f}%   over {len(have)} benchmarks")

print("\nTokens per question (prompt + completion), ours vs paper")
for m in METHODS:
    line = []
    for d in DATASETS:
        o = sum(ours[(d, m)]) / 50 if (d, m) in ours else None
        p = sum(paper[(d, m)]) / PAPER_N[d]
        line.append(f"{(f'{o:,.0f}' if o else '-'):>7s}/{p:,.0f}")
    print(f"  {LABEL.get(m, m):13s} " + "  ".join(line))

with open(R / "table2_assembled.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "method", "prompt_tokens", "completion_tokens", "accuracy", "source"])
    for d in DATASETS:
        for m in METHODS:
            if (d, m) in ours:
                w.writerow([d, m, int(ours[(d, m)][0]), int(ours[(d, m)][1]), acc[(d, m)], src[(d, m)]])
print("\nwrote result/table2_assembled.csv")
