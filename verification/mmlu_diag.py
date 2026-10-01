import json, glob, os, sys, ast, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from common import *
ROOT=sys.argv[1]; csv=load_csv(ROOT)
def parse_resp(r):
    try: v=ast.literal_eval(r) if isinstance(r,str) and r.startswith("[") else r
    except Exception: v=r
    if isinstance(v,list): v=v[0] if v else ""
    return str(v).strip()
rows=[]
for f in sorted(glob.glob(os.path.join(ROOT,"result/mmlu/mmlu_*.json"))):
    recs=json.load(open(f, encoding="utf-8")); ev=recs[-50:]
    preds=[parse_resp(r["Response"]) for r in ev]; gold=[str(r["Answer"]).strip() for r in ev]
    acc=sum(p==a for p,a in zip(preds,gold))/len(ev); bad=sum(p not in list("ABCD") for p in preds)
    best=None
    for ms in ["llama","qwen","deepseek"]:
        b=nearest_cell(csv,"MMLU",ms,stamp_of(f))
        if b is not None and (best is None or b["delta"]<best["delta"]): best=b
    tag=f"{best.model.split('/')[-1][:14]}/{best.method}" if best is not None else "?"
    ex=[(p[:12],a) for p,a in zip(preds,gold) if p!=a][:4]
    rows.append((stamp_of(f).strftime("%m-%d %H:%M"), len(recs), tag, round(acc*100), bad, str(ex)[:80]))
pd.set_option("display.width",250)
print(pd.DataFrame(rows, columns=["stamp","n_recs","cell","acc%","nonABCD","wrong (pred,gold)"]).to_string(index=False))
