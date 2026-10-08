# Table 3 (starting-graph ablation), Llama, Layered then Random, Table 1 settings.
# Runs as its own process so Claude Code's memory-pressure reaper cannot stop it.
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"
& $py -u run_cells.py --tag table3_layered --models llama --mode Layered --methods MAS_roundT AgentPrune AgentDropout --concurrency 3 --min_credit 15 *>> result\table3_run.log
& $py -u run_cells.py --tag table3_random --models llama --mode Random --methods MAS_roundT AgentPrune AgentDropout --concurrency 3 --min_credit 15 *>> result\table3_run.log
"ALL DONE $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 result\table3_run.log
