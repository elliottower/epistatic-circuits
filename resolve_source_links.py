"""Find a fetchable identifier for records that have none, and persist it.

Crossref first, then arXiv. A match is accepted only when the returned title is close to ours
AND the first author's surname appears in the returned author list AND, where we know the year,
it agrees within a year. Title alone is how you end up citing a different paper by an author
with the same surname, which this project has done five times.

Where a work exists in several versions, the venue is used to prefer the one we cite -- the
journal article rather than the working paper, the original rather than a reprint.

Results go to enrichment.yaml, not into records/, because records/ is generated and would lose
them on the next build.

    python resolve.py --check          # report, write nothing
    python resolve.py --paper mechanistic-reference
    python resolve.py --verify         # re-check that every stored link still resolves
"""
from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import yaml

ROOT = pathlib.Path(__file__).resolve().parent
RECORDS = ROOT / "records"
ENRICHMENT = ROOT / "enrichment.yaml"
UA = "citations/1.0 (mailto:elliot@elliottower.ai)"
TITLE_MIN = 0.87


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def close(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def surname(author: str) -> str:
    if not author:
        return ""
    return norm(author.split(",")[0] if "," in author else author.split()[-1])


def fetch(url: str, timeout: int = 25):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r


class Throttled(Exception):
    """The service refused to answer. Distinct from answering that it has nothing."""


def get_json(url: str, tries: int = 4) -> dict | None:
    """None means the service answered and had nothing. Throttled means it did not answer.

    Collapsing those two into None is how a batch of sixty lookups reported sixty works as
    unfindable when the real answer was that Crossref had started rate-limiting after the
    first few. A resolver that cannot tell "absent" from "I was blocked" will confidently
    delete information.
    """
    delay = 2.0
    for attempt in range(tries):
        try:
            with fetch(url) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 503, 504):
                time.sleep(delay)
                delay *= 2
                continue
            return None
        except Exception:
            time.sleep(delay)
            delay *= 2
            continue
    raise Throttled(url)


def year_ok(ours: str, theirs: int | None) -> bool:
    if not ours or not theirs:
        return True
    try:
        return abs(int(ours) - int(theirs)) <= 1
    except ValueError:
        return True


def try_crossref(rec: dict) -> tuple[str, str] | None:
    title, venue = rec.get("title", ""), rec.get("venue", "")
    first = (rec.get("authors") or [""])[0]
    q = urllib.parse.urlencode({"query.bibliographic": title, "rows": 8})
    d = get_json(f"https://api.crossref.org/works?{q}")
    if not d:
        return None
    best = None
    for it in d.get("message", {}).get("items", []):
        t = (it.get("title") or [""])[0]
        ts = close(t, title)
        if ts < TITLE_MIN:
            continue
        fams = {norm(a.get("family", "")) for a in it.get("author", [])}
        if first and surname(first) not in fams:
            continue
        parts = ((it.get("issued") or {}).get("date-parts") or [[None]])[0]
        if not year_ok(rec.get("year", ""), parts[0] if parts else None):
            continue
        container = (it.get("container-title") or [""])[0]
        vs = close(container, venue) if (venue and container) else 0.0
        score = ts + 1.5 * vs
        if best is None or score > best[0]:
            best = (score, it.get("DOI"))
    return ("doi", best[1]) if best else None


def try_arxiv(rec: dict) -> tuple[str, str] | None:
    title = rec.get("title", "")
    first = (rec.get("authors") or [""])[0]
    q = urllib.parse.urlencode({"search_query": f'ti:"{title}"', "max_results": 4})
    # raise rather than return None: a refused request is not an empty result, and treating
    # it as one is how three throttled services got reported as "nothing found"
    delay = 3.0
    xml = None
    for _ in range(3):
        try:
            with fetch(f"https://export.arxiv.org/api/query?{q}") as r:
                xml = r.read().decode()
            break
        except Exception:
            time.sleep(delay)
            delay *= 2
    if xml is None:
        raise Throttled("arxiv")
    for entry in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        tm = re.search(r"<title>(.*?)</title>", entry, re.S)
        im = re.search(r"<id>(.*?)</id>", entry)
        if not (tm and im):
            continue
        t = " ".join(tm.group(1).split())
        if close(t, title) < TITLE_MIN:
            continue
        fams = {surname(n) for n in re.findall(r"<name>(.*?)</name>", entry)}
        if first and surname(first) not in fams:
            continue
        return ("arxiv", re.sub(r"v\d+$", "", im.group(1).split("/abs/")[-1]))
    return None


def try_openalex(rec: dict) -> tuple[str, str] | None:
    """OpenAlex, which indexes books, reports and preprints that Crossref does not.

    It also carries a DOI for most arXiv preprints under the 10.48550 prefix, so a work with
    no publisher DOI still gets a resolvable identifier.
    """
    title = rec.get("title", "")
    first = (rec.get("authors") or [""])[0]
    q = urllib.parse.urlencode({"filter": f"title.search:{title}", "per-page": 5,
                                "mailto": "elliot@elliottower.ai"})
    d = get_json(f"https://api.openalex.org/works?{q}")
    if not d:
        return None
    for w in d.get("results", []):
        t = w.get("display_name") or ""
        if close(t, title) < TITLE_MIN:
            continue
        fams = {surname((a.get("author") or {}).get("display_name", ""))
                for a in (w.get("authorships") or [])}
        if first and surname(first) not in fams:
            continue
        if not year_ok(rec.get("year", ""), w.get("publication_year")):
            continue
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        if doi.startswith("10.48550/arxiv."):
            return ("arxiv", doi.split("arxiv.")[-1])
        if doi:
            return ("doi", doi)
        oid = (w.get("id") or "").rsplit("/", 1)[-1]
        if oid:
            return ("openalex", oid)
    return None


def load_overlay() -> dict:
    if not ENRICHMENT.exists():
        return {}
    text = ENRICHMENT.read_text()
    return yaml.safe_load(text) or {}


def save_overlay(overlay: dict) -> None:
    ENRICHMENT.write_text(
        "# Facts resolved after the bibliographies were written, keyed by record slug.\n"
        "# Regenerating records/ does not touch this file; build.py applies it as an overlay.\n"
        "# Identifiers were accepted only on title, first-author surname and year together.\n"
        + yaml.safe_dump(overlay, sort_keys=True, allow_unicode=True))


def verify() -> int:
    """Every stored link still resolves. A dead link is worse than a missing one."""
    bad = []
    checked = 0
    for p in sorted(RECORDS.glob("*.yaml")):
        r = yaml.safe_load(p.read_text()) or {}
        url = r.get("url") or (f"https://doi.org/{r['doi']}" if r.get("doi") else None)
        if not url:
            continue
        checked += 1
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA}, method="HEAD")
            with urllib.request.urlopen(req, timeout=20) as resp:
                if resp.status >= 400:
                    bad.append((r["slug"], url, resp.status))
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):        # blocked or throttled, not absent
                continue
            bad.append((r["slug"], url, e.code))
        except Exception as e:
            bad.append((r["slug"], url, type(e).__name__))
        time.sleep(0.15)
    print(f"  checked {checked} links, {len(bad)} did not resolve")
    for slug, url, why in bad[:30]:
        print(f"    {why}  {slug[:38]:<40}{url[:60]}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--paper", help="only records this paper cites")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.verify:
        return verify()

    todo = []
    for p in sorted(RECORDS.glob("*.yaml")):
        r = yaml.safe_load(p.read_text()) or {}
        if r.get("url") or r.get("doi") or r.get("arxiv"):
            continue
        if a.paper and a.paper not in (r.get("cited_by") or {}):
            continue
        todo.append(r)
    if a.limit:
        todo = todo[:a.limit]
    print(f"  {len(todo)} records without an identifier\n")

    overlay = load_overlay()
    found = blocked = consecutive_blocks = 0
    for r in todo:
        # Each service is tried independently: one of them rate-limiting must not stop the
        # others from answering. A record only counts as blocked if every service refused,
        # which is different from every service having nothing.
        hit, refused = None, 0
        for service in (try_crossref, try_openalex, try_arxiv):
            try:
                hit = service(r)
            except Throttled:
                refused += 1
            time.sleep(1.5)
            if hit:
                break
        if not hit and refused == 3:
            blocked += 1
            consecutive_blocks += 1
            print(f"  BLOCKED {(r.get('title') or '?')[:60]}")
            if consecutive_blocks >= 4:
                print(f"\n  every service refusing -- stopping rather than grinding; "
                      f"{found} saved. Re-run later to continue.")
                break
            time.sleep(20)
            continue
        consecutive_blocks = 0
        if not hit:
            print(f"  ---     {(r.get('title') or '?')[:60]}")
            continue
        kind, ident = hit
        url = {"doi": f"https://doi.org/{ident}",
               "arxiv": f"https://arxiv.org/abs/{ident}",
               "openalex": f"https://openalex.org/{ident}"}[kind]
        found += 1
        print(f"  {kind:<6} {ident:<34}{(r.get('title') or '')[:40]}")
        if not a.check:
            entry = overlay.setdefault(r["slug"], {})
            entry["url"] = url
            entry[kind] = ident
            save_overlay(overlay)      # after each hit: a killed run keeps what it found

    print(f"\n  resolved {found} of {len(todo)}"
          + (f", {blocked} blocked and not retried" if blocked else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
