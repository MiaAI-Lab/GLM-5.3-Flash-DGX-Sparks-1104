# SPDX-License-Identifier: Apache-2.0
"""Offline inference experiment analysis. Python 3.10+, standard library only."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Expected a finite positive measurement")
    return value


def timing_report(rows, plan):
    for key in ("ranks", "rows", "cache_modes"):
        values = plan[key]
        if not values or len(values) != len(set(values)):
            raise ValueError("Plan dimensions must be nonempty and unique")
    if type(plan["repeats"]) is not int or plan["repeats"] < 2:
        raise ValueError("At least two repeats required")
    expected = {(str(r), str(m), c, leg, str(rep))
                for r in plan["ranks"] for m in plan["rows"]
                for c in plan["cache_modes"] for leg in ("A0", "B", "A1")
                for rep in range(plan["repeats"])}
    observed = {}
    for row in rows:
        key = tuple(str(row[k]) for k in ("rank", "rows", "cache", "leg", "repeat"))
        if key in observed:
            raise ValueError("Duplicate timing coordinate: " + str(key))
        observed[key] = positive(row["ms"])
    if set(observed) != expected:
        raise ValueError(f"Timing matrix mismatch: missing={len(expected-set(observed))}, unexpected={len(set(observed)-expected)}")
    cells, groups = [], []
    for rank in plan["ranks"]:
        for cache in plan["cache_modes"]:
            sums = [0.0, 0.0, 0.0]
            for m in plan["rows"]:
                samples = [[observed[(str(rank), str(m), cache, leg, str(rep))]
                            for rep in range(plan["repeats"])] for leg in ("A0", "B", "A1")]
                med = list(map(statistics.median, samples))
                sums = [a+b for a,b in zip(sums, med)]
                cells.append(dict(rank=rank, rows=m, cache=cache, samples_ms=samples,
                                  medians_ms=med))
            groups.append(dict(rank=rank, cache=cache, sum_row_medians_ms=sums,
                               saved_vs_A0_pct=100*(1-sums[1]/sums[0]),
                               saved_vs_A1_pct=100*(1-sums[1]/sums[2]),
                               baseline_drift_pct=100*(sums[2]/sums[0]-1),
                               faster_than_both=sums[1]<min(sums[0],sums[2])))
    return dict(records=len(observed), cells=cells, groups=groups,
                local_screen_positive=all(g["faster_than_both"] for g in groups),
                scope="Equal-row sum of medians; ranks remain separate. Not a production-weighted or end-to-end estimate.")


def service_report(data):
    result = {}
    for name, legs in data.items():
        if not {"A0", "A1"}.issubset(legs) or not any(k.startswith("B") for k in legs):
            raise ValueError("Service scenario needs A0, A1 and candidate legs")
        if any(k not in {"A0", "A1"} and not k.startswith("B") for k in legs):
            raise ValueError("Unknown service leg")
        if len({len(v) for v in legs.values()}) != 1:
            raise ValueError("Unequal service repeat counts")
        med = {k: statistics.median([positive(x) for x in v]) for k,v in legs.items()}
        candidates = {k: dict(vs_A0_pct=100*(v/med["A0"]-1),
                              vs_A1_pct=100*(v/med["A1"]-1))
                      for k,v in med.items() if k.startswith("B")}
        result[name] = dict(samples_tps=legs, medians_tps=med, candidates=candidates,
                            baseline_drift_pct=100*(med["A1"]/med["A0"]-1),
                            all_candidates_above_both=all(v>max(med["A0"],med["A1"]) for k,v in med.items() if k.startswith("B")))
    if not result:
        raise ValueError("Empty service results")
    return result


def union_ms(intervals):
    ordered = sorted(intervals)
    total, end = 0.0, -math.inf
    for start, stop in ordered:
        if not all(math.isfinite(v) for v in (start,stop)) or stop < start:
            raise ValueError("Invalid interval")
        total += max(0, stop-max(start,end))
        end = max(end,stop)
    return total


def attribution(rows):
    groups = {}
    for row in rows:
        key = (row["rank"], row["request"], row["round"])
        groups.setdefault(key, []).append(row)
    output=[]
    for key, records in sorted(groups.items()):
        tokens={int(r["actual_tokens"]) for r in records}
        if len(tokens)!=1 or min(tokens)<=0:
            raise ValueError("Actual committed tokens must agree and be positive per round")
        costs={}
        for category in sorted({r["category"] for r in records}):
            intervals=[(float(r["start_ms"]),float(r["end_ms"])) for r in records if r["category"]==category]
            costs[category]=union_ms(intervals)/next(iter(tokens))
        output.append(dict(rank=key[0],request=key[1],round=key[2],actual_tokens=next(iter(tokens)),category_union_ms_per_token=costs))
    return dict(rounds=output,scope="Within-category interval unions only. Categories and ranks may overlap: do not add. Instrumented attribution is not throughput or removable time.")


def check_report(records, ranks):
    summaries=[]
    for rank in ranks:
        entries=[r for r in records if r["rank"]==rank]
        numeric=[r for r in entries if r.get("numeric")]
        controls=[r for r in entries if r.get("dependency_controls")]
        resources=[r for r in entries if r.get("resources")]
        if not numeric or not controls or len(resources)!=2:
            raise ValueError("Missing numeric, dependency or resource evidence")
        for record in numeric:
            worst=float(record["worst_tol"]); negative=float(record["negative_tol"])
            if not math.isfinite(worst) or not math.isfinite(negative) or not 0<=worst<=1 or negative<=1:
                raise ValueError("Numeric reference or negative control failed")
            if record["full_outputs"] is not True or record["bitwise_repeats"]<2:
                raise ValueError("Incomplete output comparison")
        if not all(c[k] is True for c in controls for k in ("delayed_producer","zero_input","nonfinite","restored")):
            raise ValueError("Dependency control failed")
        summaries.append(dict(rank=rank,numeric_records=len(numeric),dependency_sets=len(controls),
                              max_tolerance_fraction=max(r["worst_tol"] for r in numeric),resources=resources))
    return dict(reported_checks=summaries,scope="Validation of recorded results only; does not rerun CUDA or certify coverage, sanitizer or standard qualification.")


def analyze(case):
    plan=json.loads((case/"plan.json").read_text())
    with (case/"timings.csv").open(newline="") as f:
        local=timing_report(list(csv.DictReader(f)),plan)
    service=service_report(json.loads((case/"service.json").read_text()))
    checks=check_report(json.loads((case/"checks.json").read_text()),plan["ranks"])
    return dict(schema_version=1,local=local,service=service,recorded_checks=checks,
                end_to_end_gain_confirmed=False,
                conclusion="LOCAL_SCREEN_ONLY_NO_END_TO_END_CLAIM" if local["local_screen_positive"] else "NO_CONSISTENT_LOCAL_GAIN",
                acceptance="Directional comparisons are descriptive, not statistical or deployment acceptance. Qualification evidence is documented separately.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    a=sub.add_parser("analyze");a.add_argument("case",type=Path);a.add_argument("--output",type=Path,required=True)
    a=sub.add_parser("attribute");a.add_argument("csv",type=Path);a.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    try:
        if args.command=="analyze": result=analyze(args.case)
        else:
            with args.csv.open(newline="") as f: result=attribution(list(csv.DictReader(f)))
        if args.output.exists(): raise ValueError("Output exists; use a new run name")
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    except (ValueError,KeyError,OSError,statistics.StatisticsError) as error:
        parser.exit(2,str(error)+"\n")


if __name__=="__main__": main()
