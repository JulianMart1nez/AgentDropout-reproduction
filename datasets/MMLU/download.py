"""
Stub for the missing download helper that AgentDropout's run_mmlu.py imports.

The original AgentDropout/AgentPrune repos never shipped this module, and
their README says only "download MMLU and place it in the dataset folder"
with no script. We downloaded the standard Hendrycks MMLU test set
(https://people.eecs.berkeley.edu/~hendrycks/data.tar) and extracted it
directly to datasets/MMLU/data/{dev,val,test}/*.csv ourselves, so this
function is a no-op: it exists only so `from datasets.MMLU.download import
download` doesn't crash on import, and it verifies the expected data is
actually present rather than silently doing nothing.
"""
import os


def download():
    base = os.path.join(os.path.dirname(__file__), "data")
    for split in ("dev", "val", "test"):
        d = os.path.join(base, split)
        if not os.path.isdir(d) or not any(f.endswith(".csv") for f in os.listdir(d)):
            raise FileNotFoundError(
                f"Expected MMLU CSVs under {d}, none found. "
                "Download https://people.eecs.berkeley.edu/~hendrycks/data.tar "
                "and extract it into datasets/MMLU/ (it unpacks to data/{dev,val,test}/*.csv)."
            )
    return True
