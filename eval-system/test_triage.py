"""
Tests for the Task 5 additions: risk triage, impact tree and the CLI report.

Run from the repo root:
    python -m unittest eval-system/test_triage.py -v
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from analyzer import (  # noqa: E402
    _dig_is_stale,
    analyze_update,
    build_impact_tree,
    compute_blast_radius,
    compute_triage,
    extract_graph_features,
    render_impact_tree,
)


def toy_dig(n_direct=3, with_chain=True):
    """
    dependency 'dep' provides APIs a0..a{n-1}; function f{i} (src/main/java/p/F{i}.java) calls a{i}.
    If with_chain: test function t0 (src/test/java/p/F0Test.java) calls f0, and t1 calls t0.
    """
    nodes = [{"id": "dependency:org.x:dep", "type": "DEPENDENCY", "name": "org.x:dep"}]
    edges = []
    for i in range(n_direct):
        api, func, path = f"api:org.x.Dep.m{i}", f"function:src/main/java/p/F{i}.java::run{i}", f"src/main/java/p/F{i}.java"
        nodes += [{"id": api, "type": "API"}, {"id": func, "type": "FUNCTION"}, {"id": f"file:{path}", "type": "FILE"}]
        edges += [
            {"source": api, "target": "dependency:org.x:dep", "type": "PROVIDED_BY"},
            {"source": func, "target": api, "type": "CALLS"},
            {"source": f"file:{path}", "target": func, "type": "CONTAINS"},
        ]
    if with_chain:
        t0, t1 = "function:src/test/java/p/F0Test.java::test0", "function:src/test/java/p/F0Test.java::test1"
        for fid in (t0, t1):
            edges.append({"source": "file:src/test/java/p/F0Test.java", "target": fid, "type": "CONTAINS"})
            nodes.append({"id": fid, "type": "FUNCTION"})
        nodes.append({"id": "file:src/test/java/p/F0Test.java", "type": "FILE"})
        edges += [
            {"source": t0, "target": "function:src/main/java/p/F0.java::run0", "type": "CALLS"},
            {"source": t1, "target": t0, "type": "CALLS"},
        ]
    return {"nodes": nodes, "edges": edges}


def feats(**kw):
    base = {
        "matched_dep_nodes": ["dependency:org.x:dep"],
        "affected_functions": 5, "affected_files": 3, "direct_calls": 2, "indirect_calls": 3,
        "bump_type": "MINOR", "max_depth": 2, "func_depths": {}, "func_to_file": {},
    }
    base.update(kw)
    return base


class TriageRuleTests(unittest.TestCase):
    def test_dependency_not_in_graph_is_never_safe(self):
        t = compute_triage(feats(matched_dep_nodes=[], affected_functions=0, direct_calls=0, indirect_calls=0, affected_files=0), 0.17)
        self.assertEqual(t["score_category"], "SAFE")
        self.assertEqual(t["tier"], "RISKY")
        self.assertEqual(t["evidence"], "dependency-not-in-graph")
        self.assertEqual(t["confidence"], "low")

    def test_found_without_call_sites_major_is_risky(self):
        t = compute_triage(feats(affected_functions=0, direct_calls=0, indirect_calls=0, affected_files=0, bump_type="MAJOR"), 0.15)
        self.assertEqual(t["tier"], "RISKY")
        self.assertEqual(t["evidence"], "no-call-sites")

    def test_found_without_call_sites_patch_is_safe(self):
        t = compute_triage(feats(affected_functions=0, direct_calls=0, indirect_calls=0, affected_files=0, bump_type="PATCH"), 0.05)
        self.assertEqual(t["tier"], "SAFE")

    def test_high_score_with_call_sites_is_critical(self):
        t = compute_triage(feats(), 0.95)
        self.assertEqual((t["tier"], t["evidence"], t["confidence"]), ("CRITICAL", "call-sites-found", "high"))

    def test_mid_score_is_risky(self):
        self.assertEqual(compute_triage(feats(), 0.55)["tier"], "RISKY")

    def test_every_tier_has_next_steps_and_summary(self):
        cases = [
            feats(), feats(matched_dep_nodes=[]),
            feats(affected_functions=0, direct_calls=0, indirect_calls=0, bump_type="PATCH"),
        ]
        for f in cases:
            for score in (0.1, 0.5, 0.9):
                t = compute_triage(f, score, "org.x:dep", "1.0", "2.0")
                self.assertTrue(t["summary"])
                self.assertGreaterEqual(len(t["next_steps"]), 2)
                self.assertIn(t["tier"], ("SAFE", "RISKY", "CRITICAL"))

    def test_text_is_ascii(self):
        for score in (0.1, 0.5, 0.9):
            t = compute_triage(feats(), score, "org.x:dep", "1.0", "2.0")
            for s in [t["summary"], *t["next_steps"]]:
                s.encode("ascii")

    def test_safe_wording_is_not_a_guarantee(self):
        t = compute_triage(feats(affected_functions=0, direct_calls=0, indirect_calls=0, bump_type="PATCH"), 0.05)
        self.assertIn("not a guarantee", t["summary"])


class BlastRadiusTests(unittest.TestCase):
    def test_counts_test_functions(self):
        dig = toy_dig()
        f = extract_graph_features(dig, "org.x:dep", "1.0.0", "1.1.0")
        b = compute_blast_radius(f)
        self.assertEqual(b["functions"], 5)          # f0,f1,f2 + t0 + t1
        self.assertEqual(b["functions_in_test_files"], 2)
        self.assertEqual(b["direct_call_sites"], 3)
        self.assertEqual(b["max_depth"], 3)


class ImpactTreeTests(unittest.TestCase):
    def setUp(self):
        self.dig = toy_dig()
        self.f = extract_graph_features(self.dig, "org.x:dep", "1.0.0", "1.1.0")
        self.tree = build_impact_tree(self.dig, self.f["matched_dep_nodes"], "org.x:dep")

    def test_structure_and_depths(self):
        self.assertEqual(self.tree["total_functions"], 5)
        self.assertEqual(self.tree["shown_functions"], 5)
        first = self.tree["children"][0]
        self.assertEqual(first["name"], "F0.java::run0()")  # largest subtree first
        self.assertEqual(first["depth"], 1)
        self.assertEqual(first["children"][0]["depth"], 2)
        self.assertEqual(first["children"][0]["children"][0]["depth"], 3)
        self.assertTrue(first["children"][0]["is_test"])
        self.assertEqual(first["calls"], ["Dep.m0"])

    def test_depth_counts_match_traversal(self):
        def walk(n, acc):
            for c in n["children"]:
                acc.append(c["depth"])
                walk(c, acc)
            return acc
        from collections import Counter
        self.assertEqual(Counter(walk(self.tree, [])), Counter(self.f["func_depths"].values()))

    def test_truncation_reports_hidden(self):
        dig = toy_dig(n_direct=12, with_chain=False)
        f = extract_graph_features(dig, "org.x:dep", "1.0.0", "1.1.0")
        tree = build_impact_tree(dig, f["matched_dep_nodes"], "dep", max_children=3, max_nodes=100)
        self.assertEqual(len(tree["children"]), 3)
        self.assertEqual(tree["hidden"], 9)
        self.assertEqual(tree["shown_functions"], 3)
        self.assertIn("9 more function(s) not shown", render_impact_tree(tree))

    def test_node_budget_is_respected(self):
        dig = toy_dig(n_direct=12, with_chain=False)
        f = extract_graph_features(dig, "org.x:dep", "1.0.0", "1.1.0")
        tree = build_impact_tree(dig, f["matched_dep_nodes"], "dep", max_children=50, max_nodes=4)
        self.assertEqual(tree["shown_functions"], 4)

    def test_render_is_ascii_and_marks_tests(self):
        text = render_impact_tree(self.tree)
        text.encode("ascii")
        self.assertIn("[test]", text)
        self.assertIn("`-- ", text)
        self.assertIn("[d1, 90%]", text)

    def test_deterministic_rendering(self):
        again = build_impact_tree(self.dig, self.f["matched_dep_nodes"], "org.x:dep")
        self.assertEqual(render_impact_tree(self.tree), render_impact_tree(again))

    def test_no_dependency_gives_none(self):
        self.assertIsNone(build_impact_tree(self.dig, []))
        self.assertIn("not found", render_impact_tree(None))


class AnalyzeUpdateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # The DIG lives outside the "repo" directory: the analyzer treats <repo>/dig.json as a cached
        # DIG, which would hide the "no DIG available" cases below.
        self.repo = os.path.join(self.tmp.name, "repo")
        os.makedirs(self.repo)
        self.dig_path = os.path.join(self.tmp.name, "toy_dig.json")
        with open(self.dig_path, "w", encoding="utf-8") as fh:
            json.dump(toy_dig(), fh)

    def tearDown(self):
        self.tmp.cleanup()

    def test_graph_report_has_new_keys_and_old_schema(self):
        r = analyze_update(self.repo, "org.x:dep", "1.0.0", "2.0.0", dig_path=self.dig_path)
        for key in ("dependency", "old_version", "new_version", "risk_score", "risk_category",
                    "impacted_files", "reasoning", "recommendation"):
            self.assertIn(key, r)
        self.assertEqual(r["analysis_mode"], "graph")
        self.assertEqual(r["risk_category"], r["triage"]["tier"])
        self.assertIsNotNone(r["impact_tree"])
        self.assertIsNotNone(r["blast_radius"])
        json.dumps(r)  # must be JSON-serializable

    def test_dependency_absent_is_not_safe_and_has_no_blast_radius(self):
        r = analyze_update(self.repo, "org.nothere:missing", "1.0.0", "1.0.1", dig_path=self.dig_path)
        self.assertNotEqual(r["risk_category"], "SAFE")
        self.assertIsNone(r["blast_radius"])
        self.assertIsNone(r["impact_tree"])

    def test_empty_dig_is_treated_as_no_graph(self):
        empty = os.path.join(self.tmp.name, "empty_dig.json")
        with open(empty, "w", encoding="utf-8") as fh:
            json.dump({"nodes": [], "edges": []}, fh)
        r = analyze_update(self.repo, "org.x:dep", "1.0.0", "2.0.0", dig_path=empty)
        self.assertEqual(r["analysis_mode"], "mock")

    def test_mock_fallback_is_labelled(self):
        r = analyze_update(self.repo, "pandas", "1.0", "2.0")  # no DIG anywhere
        self.assertEqual(r["analysis_mode"], "mock")
        self.assertEqual(r["triage"]["evidence"], "simulated")
        self.assertIn("SIMULATED", r["triage"]["summary"])


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # The DIG lives outside the "repo" directory: the analyzer treats <repo>/dig.json as a cached
        # DIG, which would hide the "no DIG available" cases below.
        self.repo = os.path.join(self.tmp.name, "repo")
        os.makedirs(self.repo)
        self.dig_path = os.path.join(self.tmp.name, "toy_dig.json")
        with open(self.dig_path, "w", encoding="utf-8") as fh:
            json.dump(toy_dig(), fh)
        self.cli = os.path.join(HERE, "cli.py")

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *extra, dig=True):
        cmd = [sys.executable, self.cli, "--repo-path", self.repo, "--dependency", "org.x:dep",
               "--old-version", "1.0.0", "--new-version", "2.0.0", *extra]
        if dig:
            cmd += ["--dig-path", self.dig_path]
        return subprocess.run(cmd, capture_output=True, text=True)

    def test_text_report_sections_and_ascii(self):
        out = self.run_cli()
        self.assertEqual(out.returncode, 0, out.stderr)
        out.stdout.encode("ascii")
        for needle in ("OVERALL RISK:", "Evidence:", "Blast radius:", "Next steps:", "IMPACT TREE", "POTENTIALLY AFFECTED FILES"):
            self.assertIn(needle, out.stdout)
        for line in out.stdout.splitlines():
            if not line.startswith("  ") or "[d" not in line:   # tree lines can be long; all others fit the width
                self.assertLessEqual(len(line), 100, line)

    def test_no_tree_flag(self):
        out = self.run_cli("--no-tree")
        self.assertNotIn("IMPACT TREE", out.stdout)

    def test_json_contains_tree_and_triage(self):
        out = self.run_cli("--format", "json")
        data = json.loads(out.stdout)
        self.assertEqual(data["analysis_mode"], "graph")
        self.assertIn("next_steps", data["triage"])
        self.assertEqual(data["impact_tree"]["total_functions"], 5)

    def test_simulated_banner_in_stdout(self):
        out = self.run_cli(dig=False)
        self.assertEqual(out.returncode, 0)
        self.assertIn("SIMULATED RESULT", out.stdout)
        self.assertIn("[SIM]", out.stdout)

    def test_require_graph_exits_2_without_dig(self):
        out = self.run_cli("--require-graph", dig=False)
        self.assertEqual(out.returncode, 2)
        self.assertEqual(out.stdout.strip(), "")
        self.assertIn("--require-graph", out.stderr)

    def test_require_graph_ok_with_dig(self):
        self.assertEqual(self.run_cli("--require-graph").returncode, 0)


class StaleCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = self.tmp.name
        self.java = os.path.join(self.repo, "src", "A.java")
        os.makedirs(os.path.dirname(self.java))
        with open(self.java, "w") as fh:
            fh.write("class A {}")
        self.dig = os.path.join(self.repo, "dig.json")
        with open(self.dig, "w", encoding="utf-8") as fh:
            json.dump(toy_dig(), fh)
        now = time.time()
        os.utime(self.java, (now - 100, now - 100))   # source older than the cached DIG
        os.utime(self.dig, (now - 50, now - 50))

    def tearDown(self):
        self.tmp.cleanup()

    def touch_newer(self, path):
        t = time.time()
        os.utime(path, (t, t))

    def test_fresh_cache_is_not_stale(self):
        self.assertFalse(_dig_is_stale(self.dig, self.repo))

    def test_modified_java_makes_cache_stale(self):
        self.touch_newer(self.java)
        self.assertTrue(_dig_is_stale(self.dig, self.repo))

    def test_modified_pom_makes_cache_stale(self):
        pom = os.path.join(self.repo, "pom.xml")
        with open(pom, "w") as fh:
            fh.write("<project/>")
        self.assertTrue(_dig_is_stale(self.dig, self.repo))

    def test_ignored_directories_do_not_count(self):
        os.makedirs(os.path.join(self.repo, "target"))
        generated = os.path.join(self.repo, "target", "Gen.java")
        with open(generated, "w") as fh:
            fh.write("class Gen {}")
        self.assertFalse(_dig_is_stale(self.dig, self.repo))

    def test_missing_dig_is_not_reported_stale(self):
        self.assertFalse(_dig_is_stale(os.path.join(self.repo, "nope.json"), self.repo))

    def test_report_flags_stale_cached_dig(self):
        self.assertFalse(analyze_update(self.repo, "org.x:dep", "1.0.0", "2.0.0")["graph_stale"])
        self.touch_newer(self.java)
        r = analyze_update(self.repo, "org.x:dep", "1.0.0", "2.0.0")
        self.assertTrue(r["graph_stale"])
        self.assertIn("cached repository DIG", r["graph_source"])

    def test_explicit_dig_path_is_never_flagged(self):
        self.touch_newer(self.java)
        r = analyze_update(self.repo, "org.x:dep", "1.0.0", "2.0.0", dig_path=self.dig)
        self.assertFalse(r["graph_stale"])
        self.assertIn("explicit path", r["graph_source"])

    def test_mock_report_has_no_graph_source(self):
        with tempfile.TemporaryDirectory() as empty:
            r = analyze_update(empty, "pandas", "1.0", "2.0")
        self.assertIsNone(r["graph_source"])
        self.assertFalse(r["graph_stale"])

    def test_cli_warns_when_stale(self):
        self.touch_newer(self.java)
        cli = os.path.join(HERE, "cli.py")
        out = subprocess.run(
            [sys.executable, cli, "--repo-path", self.repo, "--dependency", "org.x:dep",
             "--old-version", "1.0.0", "--new-version", "2.0.0"],
            capture_output=True, text=True,
        )
        self.assertIn("WARNING: sources changed", out.stdout)
        self.assertIn("Graph source:", out.stdout)


if __name__ == "__main__":
    unittest.main()
