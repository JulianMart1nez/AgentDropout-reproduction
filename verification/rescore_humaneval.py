import json, glob, os, re, ast, subprocess, sys, tempfile, collections, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from common import *
ROOT=sys.argv[1]; csv=load_csv(ROOT)
OPEN = re.compile(r"```(?:python|py)?[ \t]*\n", re.I)
def extract(ans):
    if isinstance(ans, list): ans = ans[0] if ans else ""
    ans=str(ans); opens=list(OPEN.finditer(ans))
    if not opens: return ans.lstrip("`pythonPYTHON\n").rstrip("`\n"), "nofence"
    # take the last opened block that contains a def; closed or not
    for m in reversed(opens):
        body=ans[m.end():]; close=body.find("```")
        code=body[:close] if close>=0 else body
        if "def " in code: return code, ("closed" if close>=0 else "UNCLOSED")
    body=ans[opens[-1].end():]; close=body.find("```")
    return (body[:close] if close>=0 else body), ("closed" if close>=0 else "UNCLOSED")
def run(code, timeout=10):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write("from typing import *\nimport math\n"+code); path=fh.name
    try:
        r=subprocess.run([sys.executable, path], capture_output=True, timeout=timeout, text=True, errors="replace")
        if r.returncode==0: return True, "ok"
        last=[l for l in r.stderr.strip().splitlines() if l.strip()]
        return False, (last[-1][:40] if last else "nonzero")
    except subprocess.TimeoutExpired: return False, "Timeout"
    finally:
        try: os.unlink(path)
        except Exception: pass
rows=[]
for f in sorted(glob.glob(os.path.join(ROOT,"result/eval/*.json"))):
    try: recs=json.load(open(f, encoding="utf-8"))
    except Exception as e: print("SKIP malformed:", os.path.basename(f), str(e)[:60]); continue
    ev=recs[-50:]; base=os.path.basename(f); ms="llama" if "llama" in base else ("qwen" if "qwen" in base else "deepseek")
    b=nearest_cell(csv,"HumanEval",ms,stamp_of(f)); tag=b.method if b is not None else "?"
    orig=sum(str(r.get("Solved")).lower()=="true" for r in ev); new=0; reasons=collections.Counter(); kinds=collections.Counter(); lens=[]
    for r in ev:
        m_=re.search(r"['\"]entry_point['\"]\s*:\s*['\"](\w+)['\"]", str(r["Question"])); ep=m_.group(1) if m_ else ""
        raw=r.get("Attempt answer", r.get("Solution","")); raw=raw[0] if isinstance(raw,list) and raw else raw
        lens.append(len(str(raw)))
        code,kind=extract(raw); kinds[kind]+=1
        ok,why=run(code+"\n\n"+r["Tests"]+f"\n\ncheck({ep})\n") if ep else (False,"no entry_point")
        new+=ok
        if not ok: reasons[re.sub(r":.*","",why)]+=1
    rows.append(dict(stamp=stamp_of(f).strftime("%m-%d %H:%M"), model=ms, method=tag, n_recs=len(recs), orig=orig, rescored=new,
                     unclosed=kinds["UNCLOSED"], nofence=kinds["nofence"], med_len=int(sorted(lens)[len(lens)//2]), max_len=max(lens),
                     reasons=dict(reasons.most_common(3))))
    print(rows[-1], flush=True)
df=pd.DataFrame(rows); df.to_csv(os.path.join(ROOT,"verification","humaneval_rescore.csv"), index=False)
