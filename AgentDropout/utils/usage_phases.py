"""Per-phase token accounting for every run script (Team 8, Oct 2026).

run_gsm8k.py already tagged each LLM call with its phase (dec = node-dropout training,
opt = edge-dropout training, eval = evaluation). The other five scripts did not, so their
token totals mixed training with evaluation and could not be compared with the paper's
Tables 2 and 7, which count evaluation only. This module gives them the same accounting.

Usage in a script:
    from AgentDropout.utils.usage_phases import tag_task, install
    install(f"{AgentPrune_ROOT}/result/<domain>")          # once, at the start of main()
    asyncio.create_task(tag_task("eval", realized_graph.arun(...)))

At exit it prints USAGE[dec] / USAGE[opt] / USAGE[eval] / USAGE[ALL] lines and writes
usage_<stamp>.json next to the results (all calls, with tag, model, provider, tokens, cost).
"""
import atexit
import itertools
import json
import os
from datetime import datetime

from AgentDropout.llm.gpt_chat import USAGE_LOG, USAGE_TAG

_counter = itertools.count()
_result_dir = None


async def tag_task(phase, coro):
    """Run one question's graph with every LLM call inside it tagged '<phase>:<n>'."""
    USAGE_TAG.set(f"{phase}:{next(_counter)}")
    return await coro


def summary():
    phases = {}
    for u in USAGE_LOG:
        ph = (u.get("tag") or "untagged").split(":")[0]
        p = phases.setdefault(ph, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "cost_usd": 0.0})
        p["prompt_tokens"] += u["prompt_tokens"]; p["completion_tokens"] += u["completion_tokens"]; p["calls"] += 1
        p["cost_usd"] += u.get("cost") or 0.0
    total = {k: sum(p[k] for p in phases.values()) for k in ("prompt_tokens", "completion_tokens", "calls", "cost_usd")}
    return phases, total


def _report():
    if not USAGE_LOG:
        return
    phases, total = summary()
    for name, p in list(phases.items()) + [("ALL", total)]:
        print(f"USAGE[{name}]: prompt={p['prompt_tokens']} completion={p['completion_tokens']} "
              f"total={p['prompt_tokens'] + p['completion_tokens']} calls={p['calls']} cost=${p['cost_usd']:.4f}")
    if _result_dir:
        os.makedirs(_result_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        with open(os.path.join(_result_dir, f"usage_{stamp}.json"), "w", encoding="utf-8") as fh:
            json.dump({"by_phase": phases, "total": total, "calls": USAGE_LOG}, fh, indent=1)


def install(result_dir):
    global _result_dir
    if _result_dir is None:
        atexit.register(_report)
    _result_dir = str(result_dir)
