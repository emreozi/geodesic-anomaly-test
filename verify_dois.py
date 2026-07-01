#!/usr/bin/env python3
"""
verify_dois.py  --  Suggest and verify DOIs for a BibTeX file via the Crossref API.

WHY: DOIs must never be guessed. This script asks Crossref (the official DOI
registry for scholarly works) for the best match to each reference that does NOT
already carry a `doi = {...}` field, and prints a suggestion together with a
title-similarity score so you can accept or reject it by eye.

USAGE (run in YOUR environment, where api.crossref.org is reachable):

    python3 verify_dois.py refs.bib

Optional: put your e-mail in MAILTO below (Crossref's "polite pool" is faster).

OUTPUT: one block per entry ->
    KEY   [confidence]   suggested-DOI
          crossref title: ...
          bib title     : ...
Confidence = title similarity (0-1) x year-match. Review anything < 0.90.
Entries already having a DOI are skipped and reported as OK.
Entries whose type is book/inproceedings-without-DOI or that Crossref cannot
match are flagged for manual handling (some venues, e.g. old regional bulletins,
Cowles reports, ICLR/OpenReview, simply have no DOI -- leave the field out).

No third-party packages required (standard library only).
"""
import sys, re, json, time, html, difflib, urllib.parse, urllib.request

MAILTO = ""  # <- optional: "you@example.edu" for Crossref's faster polite pool
CROSSREF = "https://api.crossref.org/works"
ACCEPT_THRESHOLD = 0.90  # title similarity below this -> review manually


def parse_bib(path):
    """Very small BibTeX reader: returns list of dicts {key,type,fields{...}}."""
    txt = open(path, encoding="utf-8").read()
    entries = []
    # match @type{key, ...balanced-ish... } by scanning brace depth
    i = 0
    for m in re.finditer(r"@(\w+)\s*\{", txt):
        typ = m.group(1).lower()
        start = m.end()
        depth = 1
        j = start
        while j < len(txt) and depth > 0:
            if txt[j] == "{":
                depth += 1
            elif txt[j] == "}":
                depth -= 1
            j += 1
        body = txt[start:j - 1]
        key = body.split(",", 1)[0].strip()
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*[{\"](.*?)[}\"]\s*,?\s*(?=\w+\s*=|\Z)",
                              body, flags=re.S):
            fields[fm.group(1).lower()] = " ".join(fm.group(2).split())
        entries.append({"key": key, "type": typ, "fields": fields})
    return entries


def clean(s):
    s = re.sub(r"[{}\\]", "", s or "")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def first_author_surname(author):
    if not author:
        return ""
    first = re.split(r"\band\b", author)[0]
    first = re.sub(r"[{}\\]", "", first)
    if "," in first:
        return first.split(",")[0].strip()
    return first.strip().split()[-1] if first.strip() else ""


def crossref_query(title, author, year):
    params = {"rows": "3"}
    q = title
    if author:
        params["query.author"] = author
    params["query.bibliographic"] = q
    if MAILTO:
        params["mailto"] = MAILTO
    url = CROSSREF + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": f"verify_dois/1.0 ({MAILTO or 'anonymous'})"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    return data.get("message", {}).get("items", [])


def score(bib_title, bib_year, item):
    ct = clean(" ".join(item.get("title", []) or [""]))
    sim = difflib.SequenceMatcher(None, clean(bib_title), ct).ratio()
    yr = None
    for k in ("published-print", "published-online", "issued", "created"):
        dp = item.get(k, {}).get("date-parts", [[None]])
        if dp and dp[0] and dp[0][0]:
            yr = dp[0][0]; break
    year_ok = 1.0 if (bib_year and yr and abs(int(bib_year) - int(yr)) <= 1) else \
        (0.85 if not bib_year else 0.6)
    return sim * year_ok, ct, yr


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    entries = parse_bib(sys.argv[1])
    have = review = missing = 0
    for e in entries:
        key, f = e["key"], e["fields"]
        if f.get("doi"):
            have += 1
            print(f"OK      {key}   doi already present: {f['doi']}")
            continue
        title = f.get("title", "")
        author = first_author_surname(f.get("author", ""))
        year = f.get("year", "")
        if not title:
            print(f"SKIP    {key}   (no title field)"); continue
        try:
            items = crossref_query(title, author, year)
        except Exception as ex:
            print(f"ERROR   {key}   query failed: {ex}"); continue
        time.sleep(0.6)  # be polite to Crossref
        if not items:
            missing += 1
            print(f"NODOI?  {key}   no Crossref match -- check manually "
                  f"(book / report / OpenReview venues often have no DOI)")
            continue
        best = max((score(title, year, it) + (it,) for it in items),
                   key=lambda x: x[0])
        conf, ctitle, cyr, item = best
        doi = item.get("DOI", "")
        tag = "ACCEPT " if conf >= ACCEPT_THRESHOLD else "REVIEW "
        if conf < ACCEPT_THRESHOLD:
            review += 1
        print(f"{tag} {key}   [{conf:.2f}]  {doi}")
        print(f"          crossref: {ctitle[:80]} ({cyr})")
        print(f"          bib     : {clean(title)[:80]} ({year})")
    print("\nSummary: {} already have DOI, {} need manual review, "
          "{} no Crossref match.".format(have, review, missing))
    print("Add accepted ones to refs.bib as:  doi = {<the-doi>},")


if __name__ == "__main__":
    main()
