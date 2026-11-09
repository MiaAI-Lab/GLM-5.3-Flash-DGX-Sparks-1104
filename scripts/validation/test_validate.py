# SPDX-License-Identifier: Apache-2.0
import csv
import json
from pathlib import Path
import unittest
import validate

CASE=Path(__file__).parent/"examples/q8-scale-registers"

class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads((CASE/"plan.json").read_text())
        with (CASE/"timings.csv").open(newline="") as f:self.rows=list(csv.DictReader(f))

    def test_historical_case_does_not_promote_local_gain(self):
        report=validate.analyze(CASE)
        self.assertEqual(report["local"]["records"],504)
        self.assertTrue(report["local"]["local_screen_positive"])
        self.assertFalse(report["service"]["C1"]["all_candidates_above_both"])
        self.assertTrue(report["service"]["C4"]["all_candidates_above_both"])
        self.assertFalse(report["end_to_end_gain_confirmed"])
        self.assertAlmostEqual(report["local"]["groups"][0]["saved_vs_A0_pct"],2.756,places=3)

    def test_missing_duplicate_unexpected_and_nonfinite_rejected(self):
        for rows in (self.rows[:-1],self.rows+[self.rows[0]],self.rows+[dict(self.rows[0],rank="2")],
                     [dict(self.rows[0],ms="nan")]+self.rows[1:]):
            with self.subTest(rows=len(rows)),self.assertRaises(ValueError):
                validate.timing_report(rows,self.plan)

    def test_empty_plan_rejected(self):
        with self.assertRaises(ValueError):validate.timing_report([],dict(self.plan,ranks=[]))

    def test_negative_control_failure_rejected(self):
        records=json.loads((CASE/"checks.json").read_text())
        next(r for r in records if r.get("numeric"))["negative_tol"]=0
        with self.assertRaises(ValueError):validate.check_report(records,[0,1])

    def test_interval_union_avoids_double_counting(self):
        self.assertEqual(validate.union_ms([(1,5),(3,7),(7,9)]),8)
        with (CASE.parent/"intervals.csv").open(newline="") as f:
            report=validate.attribution(list(csv.DictReader(f)))
        self.assertEqual(report["rounds"][0]["category_union_ms_per_token"]["moe"],3)

    def test_token_denominator_mismatch_rejected(self):
        rows=[dict(rank="0",request="x",round="1",actual_tokens=str(n),category="x",start_ms="0",end_ms="1") for n in (1,2)]
        with self.assertRaises(ValueError):validate.attribution(rows)

    def test_invalid_service_measurement_rejected(self):
        with self.assertRaises(ValueError):validate.service_report({"C1":{"A0":[1,2],"B":[3,4],"A1":[0,2]}})

if __name__=="__main__":unittest.main()
