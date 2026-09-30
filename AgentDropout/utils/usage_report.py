"""
Prints one uniform "USAGE[TOTAL]: ..." line at process exit, summing whatever
this process's calls logged into AgentDropout.llm.gpt_chat.USAGE_LOG.

Every experiments/run_*.py script imports this once near the top so the
orchestrator (run_full_grid.py) can reliably parse token/cost totals from
stdout regardless of which dataset script actually ran -- run_gsm8k.py has
its own more detailed by-phase summary already; this is a second, simpler,
guaranteed line with the same format across all six scripts.
"""
import atexit


def _print_total():
    try:
        from AgentDropout.llm.gpt_chat import USAGE_LOG
    except Exception:
        return
    if not USAGE_LOG:
        return
    prompt = sum(u["prompt_tokens"] for u in USAGE_LOG)
    completion = sum(u["completion_tokens"] for u in USAGE_LOG)
    costs = [u["cost"] for u in USAGE_LOG if u.get("cost") is not None]
    cost = sum(costs) if costs else 0.0
    print(f"USAGE[TOTAL]: prompt={prompt} completion={completion} "
          f"total={prompt + completion} calls={len(USAGE_LOG)} cost=${cost:.4f}")


atexit.register(_print_total)
