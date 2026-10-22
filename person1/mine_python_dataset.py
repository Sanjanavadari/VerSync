"""Mine real merged Python dependency-update PRs; never add placeholder examples.
Green CI is a candidate label, not a verified SAFE label. Set GITHUB_TOKEN for API access."""
import csv, json, os, re, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
CSV_OUT = ROOT / "python_generalization_dataset.csv"
JSON_OUT = ROOT / "python_generalization_dataset.json"
CACHE_FILE = ROOT / "python_pr_api_cache.json"
API = "https://api.github.com"
TARGET, MAX_INSPECT = 25, 100
FIELDS = ["sample_id","ecosystem","project","repo_url","pr_url","pr_number","title",
          "dependency_name","old_version","new_version","base_sha","head_sha",
          "ci_status","ci_checks","merged_at","label","notes"]
HEADERS = {"Accept":"application/vnd.github+json","User-Agent":"VerSync-Python-Generalization-Miner",
           "X-GitHub-Api-Version":"2022-11-28"}
if os.getenv("GITHUB_TOKEN"):
    HEADERS["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]

def load_cache():
    try: return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return {}

def save_cache(cache):
    try: CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")
    except OSError: pass

def get(url, cache):
    if url not in cache:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as response:
            cache[url] = json.loads(response.read().decode("utf-8"))
        save_cache(cache)
        time.sleep(.2)
    return cache[url]

def parse_title(title):
    pats = [
        r"(?i)^(?:build\(deps(?:-dev)?\):\s*)?bump\s+(.+?)\s+from\s+([^\s]+)\s+to\s+([^\s]+)",
        r"(?i)^(?:chore\(deps\):\s*)?update\s+(?:dependency\s+)?(.+?)\s+from\s+([^\s]+)\s+to\s+([^\s]+)",
    ]
    for pat in pats:
        m = re.search(pat, title.strip())
        if m: return tuple(x.strip(" `") for x in m.groups())
    return "", "", ""

def python_dep(dep):
    return bool(dep and ":" not in dep and not re.search(
        r"(?i)(^actions/|^github/|^docker/|postcss|source-map|eslint|typescript|webpack|babel|lodash|react|vue|vite|npm|yarn|node-|@types/)", dep))

def ci_status(repo_api, sha, cache):
    try: runs = get(f"{repo_api}/commits/{sha}/check-runs?per_page=100", cache).get("check_runs", [])
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError): return "UNKNOWN", 0
    if not runs: return "NO_CHECK_RUNS", 0
    latest = {}
    for run in runs:
        name, stamp = run.get("name","unknown"), run.get("completed_at") or run.get("started_at") or ""
        if name not in latest or stamp >= latest[name][0]: latest[name] = (stamp, run)
    current = [x[1] for x in latest.values()]
    allowed = {"success","neutral","skipped"}
    if any(x.get("status") != "completed" for x in current): return "PENDING", len(current)
    if all(x.get("conclusion") in allowed for x in current) and any(x.get("conclusion") == "success" for x in current):
        return "GREEN", len(current)
    return "NOT_GREEN", len(current)

def existing_real_rows():
    if not CSV_OUT.exists(): return []
    try:
        with CSV_OUT.open(encoding="utf-8-sig", newline="") as f: old = list(csv.DictReader(f))
        return [r for r in old if "/pull/" in r.get("pr_url","") and r["pr_url"].startswith("https://github.com/")]
    except (OSError, csv.Error): return []

def main():
    cache, unique = load_cache(), {}
    for r in existing_real_rows():
        k = (r.get("pr_url",""), r.get("dependency_name",""), r.get("old_version",""), r.get("new_version",""))
        # Blank placeholder rows are excluded; retained real rows are not automatically verified.
        if k[0]: unique[k] = r
    queries = [
        'is:pr is:merged author:dependabot[bot] language:Python "from" "to"',
        'is:pr is:merged author:renovate[bot] language:Python "from" "to"',
    ]
    found = {}
    print("GitHub token:", "present" if os.getenv("GITHUB_TOKEN") else "not set (lower rate limit)")
    for query in queries:
        for page in (1,2):
            url = API + "/search/issues?" + urllib.parse.urlencode({"q":query,"per_page":50,"page":page,"sort":"updated","order":"desc"})
            try: result = get(url, cache)
            except urllib.error.HTTPError as e:
                print(f"Search failed HTTP {e.code}; set GITHUB_TOKEN and rerun.")
                break
            except (urllib.error.URLError, TimeoutError) as e:
                print("Network error:", e); break
            for item in result.get("items",[]):
                repo_api, number = item.get("repository_url",""), item.get("number")
                if repo_api and number: found[f"{repo_api}/pulls/{number}"] = item
            print("Unique PRs discovered:", len(found))
            if len(result.get("items",[])) < 50: break

    added, inspected = 0, 0
    for pr_api, item in found.items():
        if added >= TARGET or inspected >= MAX_INSPECT: break
        dep, old, new = parse_title(item.get("title",""))
        if not python_dep(dep) or not old or not new: continue
        pr_url, repo_api = item.get("html_url",""), item.get("repository_url","")
        key = (pr_url, dep, old, new)
        if not pr_url or key in unique: continue
        inspected += 1
        try:
            pr = get(pr_api, cache)
            if not pr.get("merged_at"): continue
            head, base = (pr.get("head") or {}).get("sha",""), (pr.get("base") or {}).get("sha","")
            if not head: continue
            ci, checks = ci_status(repo_api, head, cache)
            if ci != "GREEN": continue
            owner_repo = repo_api.removeprefix(API + "/repos/")
            project = owner_repo.rsplit("/",1)[-1]
            unique[key] = {
                "ecosystem":"python","project":project,"repo_url":"https://github.com/"+owner_repo,
                "pr_url":pr_url,"pr_number":pr.get("number",""),"title":item.get("title",""),
                "dependency_name":dep,"old_version":old,"new_version":new,"base_sha":base,"head_sha":head,
                "ci_status":ci,"ci_checks":checks,"merged_at":pr.get("merged_at") or "",
                "label":"GREEN_CI_CANDIDATE",
                "notes":"Real merged Python dependency PR with green head CI; pre/post builds not independently verified."
            }
            added += 1
            print(f"Added candidate {added}/{TARGET}: {project} {dep} {old}->{new}")
        except urllib.error.HTTPError as e:
            print(f"PR lookup HTTP {e.code}; stopping.")
            if e.code in (403,429): break
        except (urllib.error.URLError, TimeoutError) as e: print("Network error:", e)

    rows = list(unique.values())
    if not rows:
        print("No real PR rows available; existing outputs left unchanged.")
        return
    for i,r in enumerate(rows,1):
        r["sample_id"] = i
        r.setdefault("ecosystem","python")
    with CSV_OUT.open("w",encoding="utf-8-sig",newline="") as f:
        w = csv.DictWriter(f,fieldnames=FIELDS,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    JSON_OUT.write_text(json.dumps(rows,indent=2),encoding="utf-8")
    print("\nRows:",len(rows),"Unique PR URLs:",len({r.get("pr_url","") for r in rows if r.get("pr_url")}))
    print("Labels:",dict(Counter(r.get("label","") for r in rows)))
    print("Wrote:",CSV_OUT,JSON_OUT)

if __name__ == "__main__": main()
