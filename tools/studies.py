#!/usr/bin/env python3
"""Create, check, index and export the studies in studies/. Standard library only.

    python3 tools/studies.py new <slug> --title "Title" --track A   # next free number, from templates/study/
    python3 tools/studies.py check                                  # validate every study; exit 1 on errors
    python3 tools/studies.py index                                  # rewrite STUDIES.md
    python3 tools/studies.py renumber studies/0002-my-study         # move a study to the next free number
    python3 tools/studies.py export site/public/content             # JSON + files for the website

Front matter is one `key: value` per line with lists as [a, b]; see docs/PROTOCOL.md section 4.
"""
import argparse
import datetime
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUDIES = ROOT / "studies"
TEMPLATES = ROOT / "templates" / "study"

STATUSES = ("active", "review", "validated", "refuted", "inconclusive", "superseded")
EVIDENCE = ("E0", "E1", "E2", "E3", "E4", "E5")
TRACKS = ("A", "B", "C", "D", "E")
REQUIRED = ("id", "title", "status", "evidence", "track", "builds_on", "tags", "summary", "updated")
LIST_KEYS = ("builds_on", "tags", "authors")
DOCS = {"readme": "README.md", "review": "REVIEW.md", "notebook": "NOTEBOOK.md"}
PAPER_SECTIONS = ("Summary", "Method", "Results")   # required in README.md once a study is written up
PAGES = [("prompts", "PROMPTS.md", "Prompts"), ("contribute", "CONTRIBUTING.md", "Contribute"),
         ("roadmap", "ROADMAP.md", "Roadmap"), ("backlog", "ideas/BACKLOG.md", "Idea backlog"),
         ("protocol", "docs/PROTOCOL.md", "Research protocol"), ("compute", "docs/COMPUTE.md", "Compute budget"),
         ("agent-guide", "CLAUDE.md", "Agent guide")]
FOLDER_RE = re.compile(r"^(\d{4})-[a-z0-9]+(?:-[a-z0-9]+)*$")
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
STAMP_RE = re.compile(r"(20\d{2})(\d{2})(\d{2})_\d{6}")   # timestamps that lab.save_results puts in file names
EXPORT_MARKER = ".studies-export"
MAX_EXPORT_BYTES = 5 * 2**20


# --------------------------------------------------------------------------- parsing
def unquote(v):
    return v[1:-1] if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'" else v


def parse_front_matter(text):
    """Split the front matter from the body. Returns (meta, body, errors)."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text, ["README.md has no front matter (its first line must be ---)"]
    meta, errors = {}, []
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return meta, "\n".join(lines[i + 1:]), errors
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if not sep or not key or " " in key:
            errors.append(f"front matter line {i + 1}: expected 'key: value', got {line!r}")
        elif value.startswith("["):
            if not value.endswith("]"):
                errors.append(f"front matter '{key}': a list must be on one line, like [a, b]")
            meta[key] = [unquote(x.strip()) for x in value.strip("[]").split(",") if x.strip()]
        else:
            meta[key] = unquote(value)
    return meta, "", errors + ["front matter is not closed with ---"]


def load_studies():
    studies = []
    folders = sorted(p for p in STUDIES.iterdir() if p.is_dir()) if STUDIES.exists() else []
    for folder in folders:
        readme = folder / "README.md"
        if readme.exists():
            meta, body, errors = parse_front_matter(readme.read_text(encoding="utf-8"))
        else:
            meta, body, errors = {}, "", ["README.md is missing"]
        studies.append({"folder": folder, "meta": meta, "body": body, "errors": errors})
    return studies


def track_names():
    """Track letter -> name, read from the table in ROADMAP.md ('| A. Name | ...')."""
    names = {t: f"Track {t}" for t in TRACKS}
    roadmap = ROOT / "ROADMAP.md"
    if roadmap.exists():
        for m in re.finditer(r"^\|\s*([A-E])\.\s*([^|]+?)\s*\|", roadmap.read_text(encoding="utf-8"), re.M):
            names[m.group(1)] = m.group(2)
    return names


def open_serious_issues(review_text):
    """Numbers of blocker/major issues whose resolution is still 'open' in the current review."""
    current = review_text.split("## Earlier reviews")[0]
    found = []
    for line in current.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 5 and cells[1].lower() in ("blocker", "major") and cells[-1].lower().startswith("open"):
            found.append("#" + cells[0])
    return found


def study_files(folder):
    """Every file in a study folder except hidden ones and caches, as paths relative to the folder."""
    out = []
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder)
        if p.is_file() and not any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
            out.append(rel)
    return out


# --------------------------------------------------------------------------- check
def check_study(s, ids):
    errors, warnings = list(s["errors"]), []
    folder, meta, body = s["folder"], s["meta"], s["body"]
    m = FOLDER_RE.match(folder.name)
    if not m:
        errors.append("folder name must be NNNN-slug (lowercase letters, digits and hyphens)")
    for key in REQUIRED:
        if key not in meta:
            errors.append(f"front matter is missing '{key}'")
    for key in LIST_KEYS:
        if key in meta and not isinstance(meta[key], list):
            errors.append(f"'{key}' must be a list like [a, b]")
    if m and "id" in meta and meta["id"] != m.group(1):
        errors.append(f"id {meta['id']!r} does not match the folder number {m.group(1)}")
    for key, allowed in (("status", STATUSES), ("evidence", EVIDENCE), ("track", TRACKS)):
        if key in meta and meta[key] not in allowed:
            errors.append(f"{key} {meta[key]!r} is not one of {', '.join(allowed)}")
    deps = meta.get("builds_on") if isinstance(meta.get("builds_on"), list) else []
    for dep in deps:
        if dep == meta.get("id") or dep not in ids:
            errors.append(f"builds_on {dep!r} is not another study's id")
    if meta.get("status") == "superseded" and meta.get("superseded_by") not in ids:
        errors.append("a superseded study needs superseded_by: \"NNNN\" naming an existing study")
    try:
        updated = datetime.date.fromisoformat(meta.get("updated", ""))
    except ValueError:
        updated = None
        if "updated" in meta:
            errors.append(f"updated {meta['updated']!r} is not a date like 2026-10-29")
    if not (folder / "NOTEBOOK.md").exists():
        errors.append("NOTEBOOK.md is missing")

    review = folder / "REVIEW.md"
    review_text = review.read_text(encoding="utf-8") if review.exists() else ""
    reviewed = "## Verdict" in review_text
    if meta.get("status") == "validated":
        if not reviewed:
            errors.append("a validated study needs a review in REVIEW.md (run /review-study)")
        elif open_serious_issues(review_text):
            errors.append("validated, but REVIEW.md has open blocker or major issues: "
                          + ", ".join(open_serious_issues(review_text)))

    cited = set(re.findall(r"`((?:logs|results|archive)/[^`\s]+)`", body))
    linked = {re.split(r"[#?]", t)[0] for t in re.findall(r"\]\(([^)\s]+)\)", body)
              if not re.match(r"^(?:[a-z]+:|#)", t)}
    for ref in sorted(cited | linked):
        if ref and not (folder / ref).exists():
            errors.append(f"README refers to {ref}, which does not exist")

    headings = set(re.findall(r"^## +(.+?)\s*$", body, re.M))
    written_up = meta.get("status") not in (None, "active")
    for section in PAPER_SECTIONS:
        if section not in headings:
            (errors if written_up else warnings).append(f"README (the study's paper) has no '## {section}' section")
    if written_up and "References" not in headings:
        warnings.append("README has no '## References' section listing the works it cites")
    if "**Verdict:**" not in body:
        warnings.append("README has no **Verdict:** line")
    if len(meta.get("summary", "")) > 400:
        warnings.append("front-matter summary is over 400 characters; keep it to one sentence")
    stamps = [datetime.date(int(a), int(b), int(c)) for f in study_files(folder)
              for a, b, c in STAMP_RE.findall(f.name) if f.parts[0] in ("logs", "results")]
    if updated and stamps and max(stamps) > updated:
        warnings.append(f"results from {max(stamps)} are newer than 'updated: {updated}'; is the Summary current?")
    if meta.get("status") in ("review", "refuted", "inconclusive") and not reviewed:
        warnings.append("written up but not reviewed yet; run /review-study")
    return errors, warnings


def plural(n, word, many=None):
    return f"{n} {word}" if n == 1 else f"{n} {many or word + 's'}"


def run_check(studies, quiet=False):
    ids = {s["meta"].get("id") for s in studies}
    seen, n_err, n_warn = {}, 0, 0
    for s in studies:
        errors, warnings = check_study(s, ids)
        sid = s["meta"].get("id")
        if sid in seen:
            errors.append(f"id {sid} is also used by {seen[sid]}")
        seen.setdefault(sid, s["folder"].name)
        n_err, n_warn = n_err + len(errors), n_warn + len(warnings)
        for kind, items in (("error", errors), ("warning", warnings)):
            for msg in items:
                print(f"{s['folder'].relative_to(ROOT)}: {kind}: {msg}")
    if not quiet or n_err:
        print(f"{plural(len(studies), 'study', 'studies')} checked: "
              f"{plural(n_err, 'error')}, {plural(n_warn, 'warning')}")
    return n_err


# --------------------------------------------------------------------------- index
def cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def write_index(studies):
    lines = ["# Studies", "",
             "Generated by `python3 tools/studies.py index` from each study's front matter; don't edit by hand. "
             "Statuses and evidence levels are defined in [docs/PROTOCOL.md](docs/PROTOCOL.md).", "",
             "| ID | Study | Status | Evidence | Track | Builds on | Updated |", "|---|---|---|---|---|---|---|"]
    for s in studies:
        m, rel = s["meta"], s["folder"].relative_to(ROOT).as_posix()
        deps = ", ".join(m.get("builds_on") or []) or "–"
        lines.append(f"| {m.get('id', '?')} | [{cell(m.get('title', s['folder'].name))}]({rel}/README.md) | "
                     f"{m.get('status', '?')} | {m.get('evidence', '?')} | {m.get('track', '?')} | {deps} | "
                     f"{m.get('updated', '?')} |")
    lines += ["", "## Key results", ""]
    for s in studies:
        m = s["meta"]
        lines.append(f"- **{m.get('id', '?')} · {m.get('title', s['folder'].name)}** "
                     f"({m.get('status', '?')}, {m.get('evidence', '?')}): {m.get('summary', '')}")
    (ROOT / "STUDIES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote STUDIES.md ({plural(len(studies), 'study', 'studies')})")


# --------------------------------------------------------------------------- export
def export(studies, out_dir):
    out = Path(out_dir).resolve()
    if out.exists() and any(out.iterdir()) and not (out / EXPORT_MARKER).exists():
        sys.exit(f"refusing to replace {out}: it is not empty and was not written by this tool")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    built_on_by = {}
    for s in studies:
        for dep in s["meta"].get("builds_on") or []:
            built_on_by.setdefault(dep, []).append(s["meta"].get("id"))
    entries = []
    for s in studies:
        m, folder = s["meta"], s["folder"]
        docs = {"readme": s["body"]}
        for key, name in DOCS.items():
            if key != "readme" and (folder / name).exists():
                docs[key] = (folder / name).read_text(encoding="utf-8")
        files = []
        for rel in study_files(folder):
            if rel.as_posix() in DOCS.values():
                continue
            size = (folder / rel).stat().st_size
            copied = size <= MAX_EXPORT_BYTES
            if copied:
                dest = out / "studies" / folder.name / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(folder / rel, dest)
            files.append({"path": rel.as_posix(), "size": size, "copied": copied})
        entries.append({**{k: m.get(k) for k in REQUIRED}, "superseded_by": m.get("superseded_by"),
                        "authors": m.get("authors") or [],
                        "folder": folder.name, "built_on_by": built_on_by.get(m.get("id"), []),
                        "docs": docs, "files": files})
    pages = [{"slug": slug, "title": title, "path": path,
              "markdown": (ROOT / path).read_text(encoding="utf-8")}
             for slug, path, title in PAGES if (ROOT / path).exists()]
    index = {"generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
             "tracks": track_names(), "statuses": list(STATUSES), "evidence": list(EVIDENCE),
             "studies": entries, "pages": pages}
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    (out / EXPORT_MARKER).write_text("written by tools/studies.py export\n", encoding="utf-8")
    n_files = sum(len(e["files"]) for e in entries)
    print(f"exported {plural(len(entries), 'study', 'studies')}, {plural(len(pages), 'page')} "
          f"and {plural(n_files, 'file')} to {out}")


# --------------------------------------------------------------------------- new
def new_study(studies, slug, title, track):
    if not SLUG_RE.match(slug):
        sys.exit("slug must be lowercase letters, digits and hyphens, like exact-forgetting")
    if track not in TRACKS:
        sys.exit(f"track must be one of {', '.join(TRACKS)}")
    numbers = [int(m.group(1)) for s in studies if (m := FOLDER_RE.match(s["folder"].name))]
    sid = f"{max(numbers, default=0) + 1:04d}"
    folder = STUDIES / f"{sid}-{slug}"
    folder.mkdir(parents=True)
    today = datetime.date.today().isoformat()
    for name in DOCS.values():
        text = (TEMPLATES / name).read_text(encoding="utf-8")
        for key, value in (("{id}", sid), ("{title}", title), ("{track}", track), ("{date}", today)):
            text = text.replace(key, value)
        (folder / name).write_text(text, encoding="utf-8")
    for sub in ("logs", "results"):
        (folder / sub).mkdir()
        (folder / sub / ".gitkeep").touch()
    print(f"created {folder.relative_to(ROOT)}: fill in its README front matter and first NOTEBOOK entry")


def renumber(studies, path):
    """Move a study to the next free number, e.g. after another pull request took its number."""
    folder = (Path(path) if Path(path).exists() else STUDIES / path).resolve()
    m = FOLDER_RE.match(folder.name)
    if not m or folder.parent != STUDIES.resolve():
        sys.exit(f"{path} is not a study folder like studies/0002-my-study")
    old = m.group(1)
    others = [int(n.group(1)) for s in studies
              if s["folder"].resolve() != folder and (n := FOLDER_RE.match(s["folder"].name))]
    sid = f"{max(others, default=0) + 1:04d}"
    if sid == old:
        print(f"{folder.name} already has the next free number")
        return
    target = STUDIES / f"{sid}{folder.name[len(old):]}"
    folder.rename(target)
    headers = {"README.md": (rf'^id: *"?{old}"? *$', f'id: "{sid}"'),
               "NOTEBOOK.md": (rf"^# {old} notebook", f"# {sid} notebook"),
               "REVIEW.md": (rf"^# Review of {old}\b", f"# Review of {sid}")}
    for name, (pattern, new) in headers.items():
        doc = target / name
        if doc.exists():
            doc.write_text(re.sub(pattern, new, doc.read_text(encoding="utf-8"), count=1, flags=re.M),
                           encoding="utf-8")
    print(f"renumbered {folder.name} to {target.name}")
    users = [s["folder"].name for s in studies
             if old in (s["meta"].get("builds_on") or []) and s["folder"].resolve() != folder]
    if users:
        print(f"note: {', '.join(users)} list {old} in builds_on; check which study they mean")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pn = sub.add_parser("new", help="create the next study from templates/study/")
    pn.add_argument("slug")
    pn.add_argument("--title", required=True)
    pn.add_argument("--track", required=True, choices=TRACKS)
    sub.add_parser("check", help="validate every study")
    sub.add_parser("index", help="rewrite STUDIES.md")
    pr = sub.add_parser("renumber", help="move a study to the next free number")
    pr.add_argument("folder", help="the study folder, e.g. studies/0002-my-study")
    pe = sub.add_parser("export", help="write index.json and study files for the website")
    pe.add_argument("out_dir")
    pe.add_argument("--force", action="store_true", help="export even if the check finds errors")
    args = p.parse_args()

    studies = load_studies()
    if args.cmd == "new":
        new_study(studies, args.slug, args.title, args.track)
    elif args.cmd == "check":
        sys.exit(1 if run_check(studies) else 0)
    elif args.cmd == "index":
        write_index(studies)
    elif args.cmd == "renumber":
        renumber(studies, args.folder)
    elif args.cmd == "export":
        if run_check(studies, quiet=True) and not args.force:
            sys.exit("export stopped: fix the errors above (or pass --force)")
        export(studies, args.out_dir)


if __name__ == "__main__":
    main()
