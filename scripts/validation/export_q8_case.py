# SPDX-License-Identifier: Apache-2.0
"""Extract the historical q8 case from a user-supplied experiment root; no GPU/network access."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    if args.output.exists():parser.error("Output exists; use a new directory")
    timings=[];checks=[];sources=[]
    def load(relative):
        path=args.experiment_root/relative
        raw=path.read_bytes()
        sources.append(dict(source=relative,sha256=hashlib.sha256(raw).hexdigest()))
        return raw.decode("utf-8")
    for rank in (0,1):
        for line in load(f"q8-scale-registers/screen01/rank{rank}.log").splitlines():
            if not line.startswith("{"):continue
            v=json.loads(line)
            if v.get("timing"):
                if v["rank"]!=rank or v["variant"]!=int(v["leg"]==1):
                    raise ValueError("Timing identity mismatch")
                timings.append(dict(rank=rank,rows=v["M"],cache="scrub128MiB" if v["scrub128MiB"] else "no_forced_scrub",
                                    leg=["A0","B","A1"][v["leg"]],repeat=v["rep"],ms=v["ms"]))
            elif v.get("numeric") or v.get("dependency_controls") or v.get("resources"):
                checks.append(dict(rank=rank,**v))
    original=json.loads(load("q8-scale-register-native/confirmation-analysis.json"))
    service={}
    for scenario,legs in (("C1",["a0","b1","b2","a1"]),("C4",["c4-a0","c4-b","c4-a1"])):
        service[scenario]={leg.split("-")[-1].upper():original["legs"]["q8-scale-register-confirm-"+leg]["tps"] for leg in legs}
    plan=dict(ranks=[0,1],rows=list(range(2,9)),cache_modes=["no_forced_scrub","scrub128MiB"],repeats=6)
    from validate import timing_report,check_report,service_report
    timing_report(timings,plan);check_report(checks,plan["ranks"]);service_report(service)
    args.output.mkdir(parents=True)
    with (args.output/"timings.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["rank","rows","cache","leg","repeat","ms"])
        writer.writeheader();writer.writerows(timings)
    for name,data in (("plan",plan),("checks",checks),("service",service),("provenance",dict(
        kind="Historical measurement extract, not a fresh benchmark",sources=sources,
        transform="All timing and service tps repeats retained; numeric/resource/control records retained. Host and controller records excluded. Hashes identify local sources, not public attestation."))):
        (args.output/(name+".json")).write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")

if __name__=="__main__":main()
