import re, os, pandas as pd
from datetime import datetime
STAMP = re.compile(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}")
def stamp_of(path):
    return datetime.strptime(STAMP.search(os.path.basename(path)).group(0), "%Y-%m-%d-%H-%M-%S")
def load_csv(root):
    csv = pd.read_csv(os.path.join(root, "result/full_grid_results.csv"))
    csv["rt"] = pd.to_numeric(csv.run_time_seconds, errors="coerce")
    csv["start"] = pd.to_datetime(csv.completed_at) - pd.to_timedelta(csv.rt, unit="s")
    return csv
def nearest_cell(csv, dataset, model_substr, stamp, tol_s=900):
    c = csv[(csv.dataset==dataset) & csv.model.str.contains(model_substr, regex=False)].copy()
    if not len(c): return None
    c["delta"] = (c["start"] - stamp).abs()
    b = c.sort_values("delta").iloc[0]
    return b if b["delta"].total_seconds() < tol_s else None
