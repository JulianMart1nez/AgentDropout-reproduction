# Table 2 (Llama accuracy and eval-only tokens), Table 1 settings, FullConnected start graph.
# AgentPrune/AgentDropout on the benchmarks whose Table 1 runs lacked per-phase token logs, plus SC(CoT).
# Runs as its own process so Claude Code's memory-pressure reaper cannot stop it.
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"
& $py -u run_cells.py --tag table2 --models llama --datasets AQuA MultiArith SVAMP HumanEval --methods AgentPrune AgentDropout SC_CoT --concurrency 2 --min_credit 15 2>&1 | Out-File -Append -Encoding utf8 result\table2_run.log
& $py -u run_cells.py --tag table2 --models llama --datasets GSM8K MMLU --methods SC_CoT --concurrency 2 --min_credit 15 2>&1 | Out-File -Append -Encoding utf8 result\table2_run.log
"ALL DONE $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 result\table2_run.log
