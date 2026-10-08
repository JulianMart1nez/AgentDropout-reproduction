"""Pull the Python solution out of an LLM reply for HumanEval scoring (Team 8, Oct 2026).

The released code used  answer.lstrip("```python\\n").rstrip("\\n```").  str.lstrip removes a
SET OF CHARACTERS, not a prefix, so a reply that opens with prose ("Based on the analysis ...
```python ...") reached the executor with the prose intact and failed as a SyntaxError. DeepSeek's
decision agent always writes such prose, which is why its multi-agent HumanEval cells "collapsed"
to 42-48% and re-scored at 49-50 of 50.

Rule: take the LAST fenced block (```python, ```py or bare ```) that contains a function
definition, whether or not it is closed; if no block has one, take the last block; if there is
no fence at all, use the reply as is (minus stray backticks).
"""
import re

_OPEN = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\r?\n", re.IGNORECASE)


def extract_python_code(answer) -> str:
    if isinstance(answer, (list, tuple)):
        answer = answer[0] if answer else ""
    text = str(answer or "")
    opens = list(_OPEN.finditer(text))
    if not opens:
        return text.strip().strip("`")
    blocks = []
    for m in opens:
        body = text[m.end():]
        close = body.find("```")
        blocks.append(body[:close] if close >= 0 else body)
    for code in reversed(blocks):
        if re.search(r"^\s*def\s+\w+\s*\(", code, re.MULTILINE):
            return code.rstrip()
    return blocks[-1].rstrip()


if __name__ == "__main__":
    cases = [
        ("```python\ndef f(x):\n    return x\n```", "def f(x):"),
        ("Based on the analysis, here is the fix:\n```python\ndef f(x):\n    return x + 1\n```\nThis works.", "def f(x):"),
        ("def f(x):\n    return x\n", "def f(x):"),
        ("Draft:\n```python\ndef f(x): pass\n```\nFinal:\n```python\ndef f(x):\n    return 2\n```", "return 2"),
        ("```python\ndef f(x):\n    return x  # unclosed fence", "def f(x):"),
        ("```py\nimport math\ndef g(): return math.pi\n```", "def g()"),
        # the exact failure: lstrip would have kept the prose
        ("Python code below.\n```python\ndef h():\n    return 'ok'\n```", "def h():"),
    ]
    bad = [(t, w) for t, w in cases if w not in extract_python_code(t) or "Based on" in extract_python_code(t)
           or extract_python_code(t).lstrip().startswith(("Python code", "Draft"))]
    print("all extraction cases pass" if not bad else f"FAILED: {bad}")
