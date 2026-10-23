import csv
import json
import os
import re
import time
import urllib.request
import urllib.error
from html.parser import HTMLParser

ROOT = os.path.join("person1", "changelogs")
INDEX = os.path.join(ROOT, "changelog_index.csv")

class CommitLinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.depth = 0
        self.parts = []
        self.messages = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a":
            href = attrs.get("href", "")
            if "/commit/" in href:
                self.active = True
                self.depth = 1
                self.parts = []
            elif self.active:
                self.depth += 1
        elif self.active:
            self.depth += 1

    def handle_data(self, data):
        if self.active:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if not self.active:
            return
        if tag == "a" and self.depth == 1:
            message = " ".join(" ".join(self.parts).split())
            if message:
                self.messages.append(message)
            self.active = False
            self.depth = 0
            self.parts = []
        else:
            self.depth = max(1, self.depth - 1)

def extract_messages(html):
    parser = CommitLinkParser()
    parser.feed(html)

    ignored = {
        "view commit", "browse files", "copy full sha",
        "copy full commit", "commit", "commits"
    }
    output = []
    seen = set()

    for message in parser.messages:
        normalized = message.strip()
        if normalized.lower() in ignored:
            continue
        if re.fullmatch(r"[0-9a-fA-F]{7,40}", normalized):
            continue
        if normalized not in seen:
            seen.add(normalized)
            output.append(normalized)

    return output

with open(INDEX, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

cache = {}
updated = 0
failed = 0
skipped = 0

for row in rows:
    if row.get("evidence_status") != "LINK_ONLY":
        continue

    path = row.get("json_path", "")
    if not path or not os.path.isfile(path):
        print("SKIP missing JSON:", path)
        skipped += 1
        continue

    try:
        with open(path, encoding="utf-8-sig") as f:
            record = json.load(f)
    except Exception as exc:
        print("SKIP invalid JSON:", path, exc)
        skipped += 1
        continue

    sections = record.get("sections") or {}
    url = sections.get("github_compare_link", "")

    if not isinstance(url, str) or "/compare/" not in url:
        skipped += 1
        continue

    if url not in cache:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                html = response.read().decode("utf-8", errors="replace")
            messages = extract_messages(html)
            cache[url] = {
                "status": "EXTRACTED" if messages else "NO_COMMIT_MESSAGES",
                "commit_messages": messages,
                "http_status": 200
            }
        except urllib.error.HTTPError as exc:
            cache[url] = {
                "status": "FETCH_FAILED",
                "error": "HTTP " + str(exc.code)
            }
        except Exception as exc:
            cache[url] = {
                "status": "FETCH_FAILED",
                "error": str(exc)[:300]
            }

        time.sleep(0.4)

    evidence = cache[url]
    record["comparison_evidence"] = {
        "url": url,
        **evidence
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    if evidence["status"] == "EXTRACTED":
        updated += 1
        print("EXTRACTED:", row.get("project"), "|",
              len(evidence["commit_messages"]), "messages")
    elif evidence["status"] == "FETCH_FAILED":
        failed += 1
        print("FETCH FAILED:", row.get("project"), "|",
              evidence.get("error"))
    else:
        print("NO COMMIT TEXT:", row.get("project"))

print("\nUnique comparison URLs:", len(cache))
print("Records with extracted commit text:", updated)
print("Failed fetches:", failed)
print("Skipped records:", skipped)
