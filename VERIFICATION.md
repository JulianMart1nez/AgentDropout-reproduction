# Verification of the AgentDropout reproduction

**Repo checked:** `prajwal2021/AgentDropout-reproduction`, commit `9c8c1f4` (30 Sep 2026)
**Checked by:** Julian Martinez with Claude, 30 Sep 2026
**Checked against:** the authors' release `wangzx1219/AgentDropout` (added as `upstream` remote and diffed), the ACL 2025 paper, and the raw per-question result files in `result/`.

## Bottom line

The pipeline work is real and substantial: 105 cells ran, 16 problems were fixed, and three of those are genuine bugs in the authors' released code that belong in the midterm report. The core method code (`AgentDropout/graph/graph.py`) is untouched, so the paper/code discrepancies D1–D5 from the walkthrough still apply.

However, two of the README's headline "findings" are artifacts of answer extraction, the token/cost numbers cannot be compared to the paper as reported, and the run settings differ from the paper in nine places that the write-up does not state. Once corrected, the picture is closer to the paper than the README suggests.

| README claim | Verdict | Corrected picture |
|---|---|---|
| DeepSeek HumanEval "collapses" from 98% to 42–48% under multi-agent methods | **Artifact.** Prose-wrapped code was fed straight to the executor. | Re-scored with fenced-block extraction: MAS_round1 50/50, MAS_roundT 49/50, AgentPrune 50/50, AgentDropout 50/50 (see `verification/humaneval_rescore_2026-09-30.csv`). |
| Multi-agent methods "degrade most sharply" on MMLU (e.g. Qwen MAS_roundT 32% vs Vanilla 76%) | **Artifact.** The new answer regex's last fallback picks the first standalone letter in prose, usually the article "A". | In multi-agent MMLU cells the vote is "A" for 22–29 of 50 questions (gold has "A" for 6); 56–92% of wrong answers are "A"; only 2–14% of agent outputs contain an explicit "answer is X" marker. Needs a rerun after the regex fix. |
| Token/cost columns in `full_grid_results.csv` | **Not comparable to the paper.** They include the 40-sample training phases (dec + opt), which the paper's Table 2 excludes. | Where per-phase logs exist (GSM8K only), AgentDropout's eval-only savings reproduce for Qwen and DeepSeek (table below). |
| Single-agent Vanilla beats every multi-agent method for strong models | **Unresolved.** Partly driven by the two artifacts above; also the settings differ from the paper (temperature 0.2, majority vote, 2 rounds, pruning rate 0.25). | Re-evaluate after fixes. |

## 1. Confirmed genuine upstream bugs (report these)

These exist in the authors' release, not just in the vendored AgentPrune code. They are the strongest reproduction findings the team has.

1. **HumanEval correctness was never checked.** Upstream `experiments/run_humaneval.py` calls `PyExecutor().execute(answer, [test])`, which only `exec`s the function and the `check()` *definition*; nothing calls `check(entry_point)`. The `evaluate()` method that does call it is never used. Every submission therefore "passed". Confirmed by reading `upstream/main:experiments/run_humaneval.py` lines 167, 296, 421 and `AgentDropout/tools/coding/python_executor.py`. Consequence: the paper's HumanEval column cannot have come from the released code path as-is.
2. **MMLU answer extraction took the first character of the response** (`answer[0]`), which breaks as soon as any text precedes the letter.
3. **GSM8K numeric fallback truncated decimals and thousands separators** (`re.findall(r'\d+', pred)[-1]`), so "64.0" became "0".

Also confirmed: `temperature` and `max_tokens` were computed but never passed to the API in upstream, so upstream runs used the server's default sampling (vLLM/OpenAI default temperature is 1.0, which is what the paper reports).

## 2. Artifacts in the current results

### 2.1 HumanEval: prose around the code block

`run_humaneval.py` extracts code with `answer[0].lstrip("```python\n").rstrip("\n```")`. `lstrip` removes *characters*, not a prefix, so a response that starts with "Based on the provided information… ```python …" is passed to `exec` with the prose intact and fails with a SyntaxError. DeepSeek's multi-agent decision outputs always include such prose; Llama's and Qwen's usually do not, which is why only DeepSeek "collapsed".

Re-scoring the saved outputs with "take the last fenced block, then run `test + check(entry_point)`":

| Model / method | README (orig) | Re-scored | Paper (full 164) |
|---|---|---|---|
| DeepSeek Vanilla | 49/50 | 49/50 | 88.43 |
| DeepSeek MAS_round1 | 36/50 | 50/50 | 89.17 |
| DeepSeek MAS_roundT | 33/50 | 49/50 | 89.26 |
| DeepSeek AgentPrune | 21/50 | 50/50 | 90.91 |
| DeepSeek AgentDropout | 24/50 | 50/50 | 91.74 |
| Llama Vanilla / AgentPrune / AgentDropout | 40 / 36 / 39 | 40 / 36 / 39 | 53.33 / 51.67 / 55.84 |
| Qwen Vanilla / MAS_T / AgentPrune / AgentDropout | 49 / 48 / 50 / 47 | 49 / 48 / 49 / 47 | 85.28 / 87.08 / 86.67 / 87.92 |

Llama and Qwen are unchanged by the re-score, which shows the re-scorer is not inflating anything. One result file is malformed JSON and was skipped (`result/eval/meta-llama_llama-3.1-8b-instruct_2026-09-28-22-57-14.json`). Script: `verification/rescore_humaneval.py`.

### 2.2 MMLU: the letter-A bias

The fix in `AgentDropout/prompt/mmlu_prompt_set.py::postprocess_answer` tries "answer is X" / "Answer: X", then a leading "(X)", then `\b([A-D])\b` anywhere in the text, then the first character. Agents rarely use the explicit marker (2–14% of outputs), so most answers fall through to `\b([A-D])\b`, which matches the first standalone capital A–D in prose. That is nearly always "A" (the article, or "Option A:" when the agent restates the options). Majority voting across five such extractions then returns "A".

Evidence from the saved files (last 50 records of each multi-agent cell, gold distribution 6/8/19/17 for A/B/C/D):

| Cell (nearest match) | Acc | Predicted A/B/C/D/other | Wrong answers that are "A" |
|---|---|---|---|
| Llama MAS_round1 | 44% | 26/4/10/10/0 | 79% |
| Llama MAS_roundT | 38% | 24/5/9/8/4 | 65% |
| Qwen MAS_round1 | 48% | 25/4/11/9/1 | 81% |
| Qwen (32% cell) | 32% | 29/4/7/7/3 | 76% |
| DeepSeek MAS_round1 | 50% | 28/5/8/9/0 | 92% |
| DeepSeek MAS_roundT | 50% | 28/4/10/8/0 | 88% |

Single-agent Vanilla/CoT cells are not affected: their responses start with the letter, so the first-character path works.

Two saved records show why a better regex alone will not fix this. In one, no agent names a letter at all: the Knowledgeable Expert emits only `@entity@` search tags, the Mathematician answers "None.", and the others say "the holy is a part of what is just" in prose. In the other, three agents conclude "the correct answer is **Option A: True, True**", which the regex misses because of the bold markers and the word "Option". So the fix is in the prompt as much as the parser: require every MMLU agent to end with a line "The answer is X" (as was done for GSM8K), make the fallback search from the *end* of the text, and treat outputs with no letter as abstentions in the vote rather than letting them fall through to the first-character path. Script: `verification/mmlu_bias.py`.

### 2.3 Token counts include training

`run_full_grid.py` parses `USAGE[TOTAL]`, which `usage_report.py` prints at process exit for *all* phases. For AgentPrune and AgentDropout cells that is dec + opt + eval. The paper's Table 2/7 is evaluation only (training cost is discussed separately). This is why the CSV shows AgentDropout using 2–3× the tokens of unpruned MAS.

Per-phase logs (`result/gsm8k/usage_*.json`) exist only for GSM8K because only `run_gsm8k.py` was instrumented. Separating phases there:

| Model (GSM8K, n=50, eval only) | AgentDropout vs MAS_roundT | AgentDropout vs AgentPrune | Paper, vs AgentPrune |
|---|---|---|---|
| DeepSeek-V3-0324 | prompt −31.1%, completion −18.2% | prompt −16.0%, completion −18.4% | prompt −18.9%, completion −17.6% |
| Qwen2.5-72B | prompt −39.0%, completion −34.8% | prompt −26.1%, completion −32.6% | prompt −24.4%, completion −21.4% |
| Llama-3.1-8B | prompt −18.1%, completion +9.7% | no usage log for the AgentPrune cell | prompt −21.4%, completion −16.1% |

For Qwen and DeepSeek the paper's efficiency claim reproduces in direction and magnitude. The call counts also confirm node dropout did what the code does: 400 agent calls per 50 questions for AgentDropout versus 500 for MAS_roundT, i.e. exactly one agent removed per round (discrepancy D3).

Action: instrument the other five run scripts the same way as `run_gsm8k.py`, and add `eval_prompt_tokens` / `eval_completion_tokens` columns to the CSV. The existing per-question records for AQuA, SVAMP, MultiArith, MMLU and HumanEval carry no token fields, so eval-only numbers for those cannot be recovered without rerunning.

## 3. Settings that differ from the paper (state every one in the report)

| Setting | Paper | This repo | Where |
|---|---|---|---|
| Temperature | 1 | 0.2 (`DEFAULT_TEMPERATURE` in `AgentDropout/llm/llm.py`; fix 3.7 made it take effect) | all cells |
| Max tokens | not stated; upstream had no cap | 1000 | all cells |
| Edge dropout rate β | 0.1–0.2 | 0.25, applied twice (trigger hardcoded to batches 2 and 4) ≈ 44% of edges removed | `--pruning_rate` default, `run_*.py` |
| Decision node | FinalRefer (an LLM call) | FinalMajorVote (deterministic; no LLM call) for every MAS cell | `run_full_grid.py` |
| Rounds for HumanEval | T = 4 | 2 | `run_full_grid.py` |
| "AgentPrune" baseline | shared mask across rounds | run with `--diff` (per-round masks), i.e. the paper's Table 5 "edge-only" variant, not AgentPrune proper | `run_full_grid.py` |
| Llama model | Meta-Llama-3-8B-Instruct | meta-llama/llama-3.1-8b-instruct | README 3.2 |
| DeepSeek model | DeepSeek-V3 (Dec 2024) | deepseek-chat-v3-0324 | `run_full_grid.py` |
| Evaluation set | GSM8K 1319, HumanEval 164, MMLU 153 (others: verify in AgentPrune loaders) | first 50 of each file; MMLU first 50 of the seeded shuffle | `datasets/*_50.*` |
| Baselines | 7 for Llama incl. SC (CoT); CoT on HumanEval; AutoGen/AgentVerse on Qwen/DeepSeek | 6; CoT skipped on HumanEval; no SC (CoT) | `run_full_grid.py` |
| Serving | vLLM on A800, fp16 | OpenRouter; provider and quantization not recorded | `.env` |
| Prompts | AgentPrune's | GSM8K solver gains a mandatory "The answer is <number>" last line; CoT suffix added | `AgentDropout/prompt/gsm8k_prompt_set.py` |

Things that **do** match the paper: 40 training samples (2×20 for node dropout, 4×10 for edge dropout, hardcoded), learning rate 0.1, one node dropped per round, five agents, two rounds for math/MMLU, the AgentPrune agent configs.

## 4. Data hygiene

- `result/full_grid_results.csv` has text in numeric columns for the Llama/HumanEval/MAS_roundT cell (`~350000 (est)`). Put estimates in numeric form with a separate `estimated` flag column.
- Result files are named `*_llama3_*` for all three models, and SVAMP/MultiArith files are named `gsm8k_*`. Encode model and method in the filename, or write a manifest.
- Result files for training methods contain training records followed by eval records (90 or 130 rows); nothing marks the boundary.
- `grid.log` ends at 103/105 with two failures; the backfill logs complete them. Fine, but say so in the README.

## 5. What this means for the midterm

**Keep:** the setup write-up, the three upstream bugs, the orchestration, the Llama and Qwen accuracy numbers (subject to n=50), the GSM8K eval-only efficiency result.

**Withdraw or redo:** the DeepSeek HumanEval numbers (re-score is free, already done), all multi-agent MMLU numbers (rerun after the regex fix), the token/cost table (needs eval-only accounting), and the two "notable patterns" paragraphs.

**Before any further spending**, make these changes so nothing has to be run twice:
1. HumanEval: extract the last fenced code block; keep the `check(entry_point)` fix.
2. MMLU: force an explicit final-answer line and search from the end.
3. `temperature=1`, `--pruning_rate 0.2` (or 0.1), `--num_rounds 4` for HumanEval.
4. Decide FinalRefer vs FinalMajorVote. Matching the paper means FinalRefer; if majority vote stays, run FinalRefer on at least GSM8K to show the difference.
5. Instrument every run script for per-phase usage; add eval-only token columns.
6. Record the OpenRouter provider per call (the response carries it).
7. Add SC (CoT) with 5 samples and CoT on HumanEval.

## 6. Budget answer

Measured cost at n=50: Llama $1.01 for 35 cells, Qwen $7.42, DeepSeek $7.37. Scaling the Llama cells to the paper's evaluation-set sizes (GSM8K 1319, AQuA 254, MultiArith 180, SVAMP 300, HumanEval 164, MMLU 153) costs about **$8 per seed for all methods**. Qwen or DeepSeek at full size would be roughly $55–60 each.

The paper runs every analysis experiment (Tables 3–6, Appendix A.3/A.4) on Llama only. On Llama the five remaining analyses cost about **$2 at n=50 and roughly $25–35 at full size**, not $35–40. The $35–40 estimate only makes sense if the analyses were planned for all three models, which the paper did not do.

Recommended spend, in order:
1. $0: re-score HumanEval and fix MMLU extraction; update the README.
2. ~$25–30: Llama main grid at full set sizes, 7 methods, 3 seeds.
3. ~$2–5: all analysis tables on Llama at n=50 first, to check the pipeline.
4. ~$25–35: the analysis tables on Llama at full size (Table 5 first, then 6, 3, 4; Table 4 needs the node-count fix for rates above 0.2).
5. Only if budget remains: Qwen/DeepSeek Table 1 at 100–150 questions.

Do not skip the analyses. The course requirement is "reproduce all experimental results", and on Llama they are cheap.

## Files

- `verification/humaneval_rescore_2026-09-30.csv`: per-file re-scoring results.
- `verification/`: `rescore_humaneval.py`, `gsm8k_tokens.py`, `mmlu_bias.py`, `mmlu_diag.py`, `common.py`. Run each from the repo root as `python verification/<script>.py .` (the only argument is the repo root). `rescore_humaneval.py` executes the saved candidate solutions, so run it where that is acceptable.
- To diff against the authors' release: `git remote add upstream https://github.com/wangzx1219/AgentDropout.git && git fetch upstream && git diff --stat upstream/main HEAD -- AgentDropout experiments`.
