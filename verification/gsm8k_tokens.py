import json, glob, os, sys, collections, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from common import *
ROOT=sys.argv[1]; csv=load_csv(ROOT); rows=[]
for f in sorted(glob.glob(os.path.join(ROOT,"result/gsm8k/usage_*.json"))):
    d=json.load(open(f, encoding="utf-8")); calls=d["calls"]
    model=collections.Counter(c["model"] for c in calls).most_common(1)[0][0]
    ph=d["by_phase"]; ev=ph.get("eval",{}); b=nearest_cell(csv,"GSM8K",model,stamp_of(f))
    rows.append(dict(stamp=stamp_of(f), model=model.split("/")[-1][:18], method=b.method if b is not None else "?",
        phases="+".join(ph), eval_prompt=ev.get("prompt_tokens",0), eval_completion=ev.get("completion_tokens",0), eval_calls=ev.get("calls",0),
        train_prompt=sum(p["prompt_tokens"] for k,p in ph.items() if k!="eval"), train_completion=sum(p["completion_tokens"] for k,p in ph.items() if k!="eval"),
        csv_prompt=int(b.prompt_tokens) if b is not None and str(b.prompt_tokens).isdigit() else None))
df=pd.DataFrame(rows); pd.set_option("display.width",250); print(df.to_string(index=False))
m=df[df.method!="?"].drop_duplicates(["model","method"],keep="last").pivot_table(index="model",columns="method",values=["eval_prompt","eval_completion"])
print("\n=== GSM8K EVAL-ONLY tokens ==="); print(m.to_string())
for mod in m.index:
    p=m.loc[mod,"eval_prompt"]; c=m.loc[mod,"eval_completion"]
    try: print(f"{mod}: AgentDropout vs MAS_roundT  prompt {100*(1-p['AgentDropout']/p['MAS_roundT']):+.1f}%  completion {100*(1-c['AgentDropout']/c['MAS_roundT']):+.1f}% | vs AgentPrune prompt {100*(1-p['AgentDropout']/p['AgentPrune']):+.1f}% completion {100*(1-c['AgentDropout']/c['AgentPrune']):+.1f}% | AgentPrune vs MAS_T prompt {100*(1-p['AgentPrune']/p['MAS_roundT']):+.1f}%")
    except Exception as e: print(mod,"incomplete",e)
