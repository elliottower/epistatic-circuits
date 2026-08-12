"""One provenance record per cited work, mechval format.

Parses references.bib and paper/*.tex, writes one YAML per citation key to sources/.
Extracts verbatim quotes (``...'') and attributed claims (sentences containing \\citet
or \\citep) with file:line locators.

Depth levels:
    quoted      verbatim quote appears in the paper; quotes extracted and pinned
    cited       named via \\citet/\\citep; claims extracted but no verbatim quotes
    uncited     in the bibliography but never cited

    python build_claim_records.py --check    # report, write nothing
    python build_claim_records.py            # write to sources/
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

try:
    import yaml
except ImportError:
    print("pip install pyyaml", file=sys.stderr)
    sys.exit(1)

ROOT = pathlib.Path(__file__).resolve().parent
BIB = ROOT / "paper" / "references.bib"
PAPER_DIR = ROOT / "paper"
SOURCES = ROOT / "sources"

FIELD = re.compile(r"(\w+)\s*=\s*[{\"](.*?)[}\"]\s*,?\s*$", re.S)


def strip_tex(s: str) -> str:
    s = re.sub(r"[{}]", "", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = s.replace("\\&", "&").replace("--", "-").replace("\\", "")
    return " ".join(s.split())


def parse_bib(text: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for m in re.finditer(r"@(\w+)\s*\{([^,]+),(.*?)\n\}", text, re.S):
        kind, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)
        rec: dict[str, str] = {"type": kind}
        for line in re.split(r",\s*\n", body):
            fm = FIELD.search(line.strip())
            if fm:
                rec[fm.group(1).lower()] = strip_tex(fm.group(2))
        out[key] = rec
    return out


def cited_keys(tex_dir: pathlib.Path) -> set[str]:
    used: set[str] = set()
    for f in tex_dir.glob("*.tex"):
        text = f.read_text()
        for grp in re.findall(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}", text):
            used |= {x.strip() for x in grp.split(",") if x.strip()}
    return used


def extract_quotes(tex_dir: pathlib.Path) -> dict[str, list[dict]]:
    """Extract verbatim ``...'' quotes near citation commands.

    Handles multi-line quotes by joining the full file text first.
    """
    quotes: dict[str, list[dict]] = {}
    for f in sorted(tex_dir.glob("sec_*.tex")):
        text = f.read_text()
        lines = text.split("\n")
        comment_stripped = []
        for line in lines:
            if line.strip().startswith("%"):
                comment_stripped.append("")
            else:
                comment_stripped.append(line)
        full = "\n".join(comment_stripped)

        for m in re.finditer(r"``(.*?)''", full, re.DOTALL):
            quote_text = " ".join(m.group(1).split())
            start_pos = m.start()
            line_no = full[:start_pos].count("\n") + 1

            context_start = max(0, line_no - 6)
            context_end = min(len(lines), line_no + 6)
            context = "\n".join(lines[context_start:context_end])
            cite_matches = re.findall(r"\\cite[tp]?\{([^}]+)\}", context)
            for cite_group in cite_matches:
                for key in cite_group.split(","):
                    key = key.strip()
                    if key:
                        quotes.setdefault(key, []).append({
                            "text": quote_text,
                            "locator": f"{f.name}:{line_no}",
                            "verified": False,
                        })
    return quotes


def extract_claims(tex_dir: pathlib.Path) -> dict[str, list[dict]]:
    """Extract sentences that attribute a claim to a citation.

    Joins paragraphs into single strings so multi-line sentences are captured
    whole. A claim is any sentence containing \\citet or \\citep.
    """
    claims: dict[str, list[dict]] = {}
    cite_pat = re.compile(r"\\cite[tp]?\{([^}]+)\}")

    for f in sorted(tex_dir.glob("sec_*.tex")):
        text = f.read_text()
        lines = text.split("\n")
        non_comment = []
        line_map = []
        for i, line in enumerate(lines, 1):
            if line.strip().startswith("%"):
                continue
            non_comment.append(line)
            line_map.append(i)

        full = " ".join(non_comment)
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", full)

        for sent in sentences:
            for cm in cite_pat.finditer(sent):
                keys = [k.strip() for k in cm.group(1).split(",") if k.strip()]
                display = re.sub(r"\\cite[tp]?\{[^}]+\}", "[CITE]", sent)
                display = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", display)
                display = re.sub(r"[{}$]", "", display)
                display = " ".join(display.split())
                if len(display) < 15:
                    continue
                first_word = sent[:40]
                line_no = 0
                for idx, line in enumerate(non_comment):
                    if first_word[:20] in line:
                        line_no = line_map[idx]
                        break
                for key in keys:
                    claims.setdefault(key, []).append({
                        "claim": display,
                        "locator": f"{f.name}:{line_no}",
                    })
                break
    return claims


def authors(raw: str) -> list[str]:
    if not raw:
        return []
    return [" ".join(a.split()) for a in re.split(r"\s+and\s+", raw) if a.strip()]


def venue(rec: dict) -> str:
    for k in ("booktitle", "journal", "howpublished", "school", "publisher"):
        if rec.get(k):
            return rec[k]
    return ""


def identifier(rec: dict) -> dict:
    out = {}
    if rec.get("url"):
        out["url"] = rec["url"]
    if rec.get("doi"):
        out["doi"] = rec["doi"]
    note = rec.get("note", "")
    m = re.search(r"arXiv:(\d{4}\.\d{4,5})", note)
    if m:
        out.setdefault("arxiv", m.group(1))
        out.setdefault("url", f"https://arxiv.org/abs/{m.group(1)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="Report what would be written without writing")
    a = ap.parse_args()

    bib = parse_bib(BIB.read_text())
    used = cited_keys(PAPER_DIR)
    all_quotes = extract_quotes(PAPER_DIR)
    all_claims = extract_claims(PAPER_DIR)

    counts = {"quoted": 0, "cited": 0, "uncited": 0}
    for key, rec in sorted(bib.items()):
        has_quotes = key in all_quotes
        depth = ("quoted" if has_quotes
                 else "cited" if key in used
                 else "uncited")
        counts[depth] += 1

        if a.check:
            n_quotes = len(all_quotes.get(key, []))
            n_claims = len(all_claims.get(key, []))
            print(f"  {depth:8s}  {key:40s}  quotes={n_quotes}  claims={n_claims}")
            continue

        doc = {
            "citation": key,
            "depth": depth,
            "title": rec.get("title", ""),
            "authors": authors(rec.get("author", "")),
            "year": rec.get("year", ""),
            "venue": venue(rec),
            **identifier(rec),
        }

        if has_quotes:
            doc["quotes"] = all_quotes[key]

        if key in all_claims:
            doc["attributed_claims"] = all_claims[key]

        SOURCES.mkdir(exist_ok=True)
        (SOURCES / f"{key}.yaml").write_text(
            yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100))

    print(f"\n  quoted (verbatim quotes in paper)      {counts['quoted']:>4}")
    print(f"  cited (\\citet/\\citep in paper)          {counts['cited']:>4}")
    print(f"  in bib but never cited                 {counts['uncited']:>4}")
    print(f"  {'─' * 42}")
    print(f"  total records                          {sum(counts.values()):>4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
