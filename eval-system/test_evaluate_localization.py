"""
Unit tests for evaluate_localization.py (standard library only).

Run from the repo root:
    python -m unittest eval-system/test_evaluate_localization.py -v
or from eval-system/:
    python -m unittest test_evaluate_localization -v
"""

import json
import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_localization import (  # noqa: E402
    bootstrap_mean_ci,
    diagnose_missing_dependency,
    evaluate_ranking,
    expected_random_metrics,
    hit_sets,
    metrics_from_hits,
    normalize_path,
    paths_match,
    rank_files,
    rank_files_as_shipped,
)


class PathMatchingTests(unittest.TestCase):
    def test_exact_and_suffix(self):
        self.assertTrue(paths_match("mod/src/main/java/a/B.java", "mod/src/main/java/a/B.java"))
        self.assertTrue(paths_match("mod/src/main/java/a/B.java", "a/B.java"))  # gt is a suffix
        self.assertTrue(paths_match("a/B.java", "mod/src/main/java/a/B.java"))  # pred is a suffix

    def test_no_partial_segment_or_basename_match(self):
        self.assertFalse(paths_match("x/aB.java", "B.java"))      # 'aB.java' is not a segment suffix
        self.assertFalse(paths_match("x/y/B.java", "z/B.java"))   # same basename, different dirs
        self.assertFalse(paths_match("", "a/B.java"))

    def test_normalization(self):
        self.assertEqual(normalize_path(".\\a\\B.java"), "a/B.java")
        self.assertTrue(paths_match("a/b/C.java", "a\\b\\C.java"))


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.gt = ["a/B.java", "c/D.java"]
        self.ranked = ["x/Y.java", "src/a/B.java", "c/D.java", "z/Z.java"]

    def test_known_values(self):
        m = evaluate_ranking(self.ranked, self.gt)
        self.assertEqual(m["hit@1"], 0.0)
        self.assertEqual(m["hit@3"], 1.0)
        self.assertAlmostEqual(m["precision@3"], 2 / 3)
        self.assertAlmostEqual(m["precision@1"], 0.0)
        self.assertAlmostEqual(m["recall@1"], 0.0)
        self.assertAlmostEqual(m["recall@3"], 1.0)
        self.assertAlmostEqual(m["recall@5"], 1.0)
        self.assertAlmostEqual(m["rr"], 0.5)  # first hit at rank 2

    def test_precision_divides_by_k_even_for_short_rankings(self):
        m = evaluate_ranking(["a/B.java"], ["a/B.java"])
        self.assertEqual(m["hit@10"], 1.0)
        self.assertAlmostEqual(m["precision@10"], 0.1)
        self.assertAlmostEqual(m["precision@1"], 1.0)
        self.assertAlmostEqual(m["rr"], 1.0)

    def test_empty_ranking_scores_zero(self):
        m = evaluate_ranking([], self.gt)
        self.assertTrue(all(v == 0.0 for v in m.values()))

    def test_no_hit(self):
        m = evaluate_ranking(["q/R.java", "s/T.java"], self.gt)
        self.assertEqual(m["hit@10"], 0.0)
        self.assertEqual(m["rr"], 0.0)

    def test_duplicate_matches_do_not_inflate_recall(self):
        # two predicted files match the same ground-truth file
        m = evaluate_ranking(["m1/a/B.java", "m2/a/B.java"], ["a/B.java", "c/D.java"])
        self.assertAlmostEqual(m["recall@5"], 0.5)

    def test_metrics_from_hits_matches_evaluate_ranking(self):
        hits = hit_sets(self.ranked, self.gt)
        self.assertEqual(metrics_from_hits(hits, len(self.gt)), evaluate_ranking(self.ranked, self.gt))


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.func_depths = {
            "function:f1": 1,  # file A, direct
            "function:f2": 1,  # file A, direct
            "function:f3": 1,  # file B, direct
            "function:f4": 2,  # file C, transitive
            "function:f5": 2,  # file D, transitive
        }
        self.func_to_file = {
            "function:f1": "src/A.java",
            "function:f2": "src/A.java",
            "function:f3": "src/B.java",
            "function:f4": "src/C.java",
            "function:f5": "src/D.java",
        }

    def test_rank_files_order_and_tiebreak(self):
        # A (2 direct) > B (1 direct) > C, D (depth 2; equal stats -> alphabetical)
        self.assertEqual(
            rank_files(self.func_depths, self.func_to_file),
            ["src/A.java", "src/B.java", "src/C.java", "src/D.java"],
        )

    def test_rank_files_is_deterministic_under_input_order(self):
        reordered = dict(reversed(list(self.func_depths.items())))
        self.assertEqual(
            rank_files(reordered, self.func_to_file),
            rank_files(self.func_depths, self.func_to_file),
        )

    def test_shipped_order_dedupes_files_and_puts_direct_callers_first(self):
        shipped = rank_files_as_shipped(self.func_depths, self.func_to_file)
        self.assertEqual(len(shipped), len(set(shipped)))
        self.assertEqual(set(shipped[:2]), {"src/A.java", "src/B.java"})

    def test_file_recovered_from_function_id_when_unmapped(self):
        depths = {"function:src/E.java::run": 1}
        self.assertEqual(rank_files(depths, {}), ["src/E.java"])

    def test_empty(self):
        self.assertEqual(rank_files({}, {}), [])
        self.assertEqual(rank_files_as_shipped({}, {}), [])


class BaselineTests(unittest.TestCase):
    def test_random_expectation_matches_analytic_value(self):
        # 4 files, exactly 1 matches the single ground-truth file: E[Hit@1] = 1/4, E[Hit@4] = 1
        hits = hit_sets(["a/A.java", "b/B.java", "c/C.java", "d/D.java"], ["c/C.java"])
        m = expected_random_metrics(hits, 1, shuffles=4000, seed=1)
        self.assertAlmostEqual(m["hit@1"], 0.25, delta=0.03)
        self.assertAlmostEqual(m["hit@5"], 1.0)
        self.assertAlmostEqual(m["recall@5"], 1.0)

    def test_random_is_seeded(self):
        hits = hit_sets(["a/A.java", "b/B.java", "c/C.java"], ["b/B.java"])
        self.assertEqual(
            expected_random_metrics(hits, 1, 50, 7), expected_random_metrics(hits, 1, 50, 7)
        )

    def test_random_empty_pool(self):
        self.assertEqual(expected_random_metrics([], 2, 10, 0)["hit@1"], 0.0)

    def test_bootstrap_ci_brackets_mean(self):
        vals = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        lo, hi = bootstrap_mean_ci(vals, 500)
        self.assertLessEqual(lo, sum(vals) / len(vals))
        self.assertGreaterEqual(hi, sum(vals) / len(vals))

    def test_bootstrap_ci_needs_two_samples(self):
        lo, hi = bootstrap_mean_ci([1.0], 100)
        self.assertNotEqual(lo, lo)  # NaN


class TraversalDeterminismTests(unittest.TestCase):
    """analyzer.traverse_impacted_callsites must not depend on Python's string hash seed."""

    SNIPPET = (
        "import json, sys; sys.path.insert(0, {here!r});"
        "from analyzer import traverse_impacted_callsites as t;"
        "apis = ['api:%d' % i for i in range(12)];"
        "nodes = [{{'id': 'dep', 'type': 'DEPENDENCY'}}];"
        "edges = [{{'source': a, 'target': 'dep', 'type': 'PROVIDED_BY'}} for a in apis];"
        "edges += [{{'source': 'function:f%d' % i, 'target': apis[i], 'type': 'CALLS'}} for i in range(12)];"
        "edges += [{{'source': 'file:src/F%d.java' % i, 'target': 'function:f%d' % i, 'type': 'CONTAINS'}} for i in range(12)];"
        "depths, _, _ = t({{'nodes': nodes, 'edges': edges}}, ['dep']);"
        "print(json.dumps(list(depths)))"
    )

    def _order(self, seed: str):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        code = self.SNIPPET.format(here=os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        return json.loads(out.stdout)

    def test_function_order_is_identical_across_hash_seeds(self):
        orders = [self._order(seed) for seed in ("1", "2", "3", "4", "5")]
        self.assertEqual(len(orders[0]), 12)
        for other in orders[1:]:
            self.assertEqual(orders[0], other)


class DiagnosisTests(unittest.TestCase):
    DIG = {
        "nodes": [
            {"id": "api:io.dropwizard.core.Application.run", "type": "API"},
            {"id": "api:io.dropwizard.core.setup.Bootstrap.add", "type": "API"},
            {"id": "api:org.other.Thing.go", "type": "API"},
            {"id": "dependency:io.dropwizard:dropwizard-core", "type": "DEPENDENCY"},
        ],
        "edges": [
            {"source": "api:io.dropwizard.core.Application.run", "target": "dependency:io.dropwizard:dropwizard-core", "type": "PROVIDED_BY"},
            {"source": "api:io.dropwizard.core.setup.Bootstrap.add", "target": "dependency:io.dropwizard:dropwizard-core", "type": "PROVIDED_BY"},
        ],
    }

    def test_never_called(self):
        reason, n, resolved = diagnose_missing_dependency("com.h2database", self.DIG)
        self.assertEqual((n, resolved), (0, []))
        self.assertIn("never called", reason)

    def test_resolved_to_sibling_artifact(self):
        reason, n, resolved = diagnose_missing_dependency("io.dropwizard", self.DIG)
        self.assertEqual(n, 2)
        self.assertEqual(resolved, ["io.dropwizard:dropwizard-core"])
        self.assertIn("different artifact", reason)

    def test_group_prefix_must_match_a_whole_segment(self):
        reason, n, _ = diagnose_missing_dependency("io.drop", self.DIG)  # not a segment boundary
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()
