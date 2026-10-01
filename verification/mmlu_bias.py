import json, glob, os, sys, ast, re, collections, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from common import *
ROOT=sys.argv[1]
def parse_resp(r):
    try: v=ast.literal_eval(r) if isinstance(r,str) and r.startswith("[") else r
    except Exception: v=r
    if isinstance(v,list): v=v[0] if v else ""
    return str(v).strip()
MARK = re.compile(r"(?:answer is|Answer:|final answer)\s*[:\-]?\s*\(?([A-D])\)?", re.I)
print("file  n  acc%  pred-dist(A/B/C/D/other)  gold-dist  wrong->A%  agents-with-marker%")
shown=0
for f in sorted(glob.glob(os.path.join(ROOT,"result/mmlu/mmlu_*.json"))):
    recs=json.load(open(f, encoding="utf-8")); ev=recs[-50:]
    preds=[parse_resp(r["Response"]) for r in ev]; gold=[str(r["Answer"]).strip() for r in ev]
    if any(len(p)>1 for p in preds): continue   # skip single-agent files (raw text responses)
    pd_=collections.Counter(preds); gd=collections.Counter(gold)
    wrong=[(p,a,r) for p,a,r in zip(preds,gold,ev) if p!=a]
    wrongA=sum(1 for p,a,r in wrong if p=="A")/max(len(wrong),1)
    # how many per-agent outputs contain an explicit marker?
    tot=hit=0
    for r in ev:
        try: aa=ast.literal_eval(r["All_answers"]) if isinstance(r["All_answers"],str) else r["All_answers"]
        except Exception: aa=[]
        for d in aa:
            for k,v in (d.items() if isinstance(d,dict) else []):
                txt=" ".join(v) if isinstance(v,list) else str(v); tot+=1; hit+=bool(MARK.search(txt))
    acc=sum(p==a for p,a in zip(preds,gold))/50
    print(f"{stamp_of(f):%m-%d %H:%M} 50 {acc*100:4.0f}  {pd_['A']}/{pd_['B']}/{pd_['C']}/{pd_['D']}/{50-sum(pd_[x] for x in 'ABCD')}  {gd['A']}/{gd['B']}/{gd['C']}/{gd['D']}  {wrongA*100:5.0f}  {100*hit/max(tot,1):5.0f}")
    if shown<2 and wrong:
        p,a,r=wrong[0]; shown+=1
        print("   EXAMPLE wrong record: pred",p,"gold",a)
        try: aa=ast.literal_eval(r["All_answers"])
        except Exception: aa=[]
        for d in aa[:5]:
            for k,v in d.items():
                txt=(" ".join(v) if isinstance(v,list) else str(v)).replace("\n"," ")
                m=MARK.search(txt); i=txt.lower().rfind("answer")
                print(f"      {k[:22]:22s} marker={m.group(1) if m else '-'} | ...{txt[max(0,i-60):i+60]!r}")
