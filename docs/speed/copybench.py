#!/usr/bin/env python3
"""Copy-heavy decode: each stream gets a fixed Python file and returns it whole with one function renamed.

The reply repeats its prompt almost token for token, which is what context copies verify (patch 0115 lets a copy
take up to 31 tokens a round instead of 15). Greedy, thinking off, fixed bytes in every leg, so token_sha must match
between legs and only the round count may move. Reports per-request decode tok/s (prefill removed), tokens a round
and wall-clock aggregate, and writes docs/speed/copy-<label>.json.
  copybench.py <label>        env: U (server), LOADS (1,4), TRIALS (2), MAXTOK (2400)
"""
import json, os, statistics, sys, threading, time, urllib.request

BASE = os.environ.get("U", "http://127.0.0.1:8888")
MODEL = os.environ.get("M", "GLM-5.3-Flash")
LOADS = [int(x) for x in os.environ.get("LOADS", "1,4").split(",")]
TRIALS = int(os.environ.get("TRIALS", "2"))
MAXTOK = int(os.environ.get("MAXTOK", "2400"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def source(k):
    """A deterministic ~170-line module; k varies names and constants so four streams copy four different files."""
    lines = [f'"""Inventory helpers, module {k}."""', "import math", "from dataclasses import dataclass", ""]
    for i in range(12):
        n = f"item_{k}_{i}"
        lines += [
            "@dataclass",
            f"class Record{i}:",
            "    name: str",
            "    count: int",
            f"    weight: float = {1.5 + i + k / 10:.2f}",
            "",
            f"def load_{n}(rows):",
            f'    """Parse rows into Record{i} values, skipping blanks."""',
            "    out = []",
            "    for row in rows:",
            "        if not row.strip():",
            "            continue",
            "        name, count = row.split(',')",
            f"        out.append(Record{i}(name.strip(), int(count) + {i}))",
            "    return out",
            "",
        ]
    lines += ["def total_weight(records):", "    return math.fsum(r.weight * r.count for r in records)", ""]
    return "\n".join(lines)


def chat(text, maxtok):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": text}], "temperature": 0.0,
                       "max_tokens": maxtok, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    tf, sp, u = r.get("tensorfold", {}), r.get("speculative", {}), r.get("usage", {})
    return {"wall": time.time() - t0, "completion": u.get("completion_tokens"), "seconds": tf.get("seconds"),
            "prefill_s": tf.get("prefill_seconds"), "token_sha": tf.get("token_sha"), "spec": sp}


def prompt(k):
    return ("Here is a Python file:\n\n```python\n" + source(k) + "\n```\n\nRename the function total_weight to "
            "sum_weights. Return the complete file with only that change, in one python code block, nothing else.")


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    res = {}
    print(f"# copy bench {label}, {MODEL} at {BASE}, {TRIALS} trials, max_tokens {MAXTOK}")
    for n in LOADS:
        trials = []
        for t in range(TRIALS):
            outs, errs = [None] * n, []

            def run(i):
                try:
                    outs[i] = chat(prompt(i), MAXTOK)
                except Exception as e:
                    errs.append(repr(e)[:160])
            th = [threading.Thread(target=run, args=(i,)) for i in range(n)]
            t0 = time.time()
            for x in th: x.start()
            for x in th: x.join()
            el = time.time() - t0
            if errs or not all(outs):
                print(f"  {n} stream(s) trial {t + 1}: ERROR {errs[:1]}")
                continue
            dtps = [o["completion"] / ((o["seconds"] or 0) - (o["prefill_s"] or 0)) for o in outs]
            tpr = [(o["spec"] or {}).get("tokens_per_round") for o in outs]
            agg = sum(o["completion"] for o in outs) / el
            shas = [o["token_sha"] for o in outs]
            trials.append({"agg": agg, "dtps": dtps, "tpr": tpr, "sha": shas, "tokens": [o["completion"] for o in outs]})
            print(f"  {n} stream(s) trial {t + 1}: aggregate {agg:6.1f} tok/s  per-stream decode "
                  f"{statistics.mean(dtps):6.1f}  tok/round {statistics.mean([x for x in tpr if x] or [0]):.2f}  "
                  f"tokens {[o['completion'] for o in outs]}  sha {[s[:12] if s else s for s in shas]}")
        res[str(n)] = trials
    p = os.path.join(OUT, f"copy-{label}.json")
    json.dump(res, open(p, "w"), indent=1)
    print(f"# wrote {p}")


if __name__ == "__main__":
    main()
