"""Check that reframing a post (SPEC-post-voice.md) left its evidence alone, against scripts/voice-baseline.json.

Usage, from the repo root:  python3 scripts/voice_check.py [--final]     compare every post with the baseline
                            python3 scripts/voice_check.py --baseline    write the baseline once, before any edit
A post is a topic source (tools/posts/src/<slug>.md) or a long-form page (posts/<slug>/index.html with no source).
Code blocks, tables, inline code, maths, headings, front matter and links must stay byte-identical (inline spans
as a multiset, so a sentence may move). Prose may change, but every number in it must already be in the post, none
may vanish, and it may grow by at most 10%. --final also fails a post whose prose did not change at all.
Output names posts, rules and numbers only, never prose, so it cannot print a private term.
"""
import hashlib
import html
import json
import re
import sys
from decimal import Decimal
from pathlib import Path

BASELINE = "scripts/voice-baseline.json"
WORDS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}
WORDS.update(twenty="20", thirty="30", forty="40", fifty="50", sixty="60", seventy="70", eighty="80", ninety="90",
             hundred="100", thousand="1000", million="1000000", billion="1000000000")
NUMBER = re.compile(r"(?<![\d.])\d+(?:[,_]\d{3})*(?:\.\d+)?|\b(?:" + "|".join(WORDS) + r")\b", re.I)
MATHS = re.compile(r"\\\(.*?\\\)|\\\[.*?\\\]", re.S)


def digest(parts):
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()


def numbers(text):
    out = set()
    for m in NUMBER.findall(text):
        word = WORDS.get(m.lower())
        out.add(word or format(Decimal(re.sub("[,_]", "", m)).normalize(), "f"))
    return out


def spans(rx, text):
    """Pull rx's matches out of text: (sorted matches, whitespace-normalized, so rewrapping a line is free; rest)."""
    found = [" ".join(s.split()) for s in rx.findall(text)]
    return sorted(found), rx.sub(" ", text)


def markdown(text):
    """Split a source the way tools/leetcode/build.py renders it: front matter, then line-led blocks, then prose."""
    _, front, body = text.split("---\n", 2)
    code, tables, display, heads, prose = [], [], [], [], []
    lines, i = body.split("\n"), 0
    while i < len(lines):
        line, j = lines[i], i + 1
        if line.startswith("```"):
            while not lines[j].startswith("```"):
                j += 1
            code.append("\n".join(lines[i:j + 1]))
            j += 1
        elif line.startswith("\\["):
            j = i
            while "\\]" not in lines[j]:
                j += 1
            display.append(" ".join("\n".join(lines[i:j + 1]).split()))
            j += 1
        elif line.startswith("|"):
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            tables.append("\n".join(lines[i:j]))
        elif re.match(r"#{2,3} ", line):
            heads.append(line)
        else:
            prose.append(line)
        i = j
    text = "\n".join(prose)
    inline, text = spans(re.compile(r"`[^`]+`"), text)
    maths, text = spans(MATHS, text)
    links = sorted(re.findall(r"\]\(([^)\s]+)\)", text))
    text = re.sub(r"\]\([^)\s]+\)", "]", text)
    return code, tables, inline, sorted(display + maths), [front] + heads + links, text


def page(text):
    """Split a long-form page's <article>, leaving out its header, nav and aside (reading-shell's markup)."""
    art = text[text.index("<article"):text.index("</article>")]
    art = re.sub(r"<(header|nav|aside)\b.*?</\1>", " ", art, flags=re.S)
    code, art = re.findall(r"<pre\b.*?</pre>", art, re.S), re.sub(r"<pre\b.*?</pre>", " ", art, flags=re.S)
    tables, art = re.findall(r"<table\b.*?</table>", art, re.S), re.sub(r"<table\b.*?</table>", " ", art, flags=re.S)
    heads, art = re.findall(r"<h[1-6]\b.*?</h[1-6]>", art, re.S), re.sub(r"<h[1-6]\b.*?</h[1-6]>", " ", art, flags=re.S)
    inline, art = spans(re.compile(r"<code\b.*?</code>", re.S), art)
    links = sorted(re.findall(r'\b(?:href|id)="([^"]*)"', art))
    maths, prose = spans(MATHS, html.unescape(re.sub(r"<[^>]+>", " ", art)))
    return code, tables, inline, maths, heads + links, prose


def posts(root):
    srcs = sorted(root.glob("tools/posts/src/*.md"))
    stems = {p.stem for p in srcs}
    longs = sorted(p for p in root.glob("posts/*/index.html") if p.parent.name not in stems)
    return srcs + longs


def snapshot(root):
    snap = {}
    for path in posts(root):
        text = path.read_text()
        code, tables, inline, maths, frozen, prose = (markdown if path.suffix == ".md" else page)(text)
        snap[path.relative_to(root).as_posix()] = {
            "code": digest(code), "tables": digest(tables), "inline": digest(inline), "maths": digest(maths),
            "frozen": digest(frozen), "prose": digest(prose.split()), "words": len(prose.split()),
            "numbers": sorted(numbers(prose), key=Decimal),
            "evidence": sorted(numbers("\n".join(code + tables + inline + maths)), key=Decimal)}
    return snap


def write_baseline(root):
    out = root / BASELINE
    if out.exists():
        sys.exit(f"{BASELINE} exists; it is written once, from main before the first edit")
    out.write_text(json.dumps(snapshot(root), indent=1, sort_keys=True) + "\n")


RULES = [("code", "code blocks changed"), ("tables", "tables changed"), ("inline", "inline code changed"),
         ("maths", "maths changed"), ("frozen", "headings, front matter or links changed")]


def check(root, final=False):
    base, now, errors = json.loads((root / BASELINE).read_text()), snapshot(root), []
    errors += [f"{p}: missing" for p in sorted(set(base) - set(now))]
    errors += [f"{p}: not in the baseline" for p in sorted(set(now) - set(base))]
    for p in sorted(set(base) & set(now)):
        b, n = base[p], now[p]
        errors += [f"{p}: {rule}" for key, rule in RULES if b[key] != n[key]]
        new = set(n["numbers"]) - set(b["numbers"]) - set(b["evidence"])
        gone = set(b["numbers"]) - set(n["numbers"])
        if new:
            errors.append(f"{p}: new numbers in prose: " + ", ".join(sorted(new, key=Decimal)))
        if gone:
            errors.append(f"{p}: numbers dropped from prose: " + ", ".join(sorted(gone, key=Decimal)))
        if n["words"] * 10 > b["words"] * 11:
            errors.append(f"{p}: prose grew {b['words']} -> {n['words']} words (limit 10%)")
        if final and b["prose"] == n["prose"]:
            errors.append(f"{p}: prose unchanged")
    return errors


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    if sys.argv[1:] == ["--baseline"]:
        write_baseline(root)
        sys.exit(0)
    if sys.argv[1:] not in ([], ["--final"]):
        sys.exit(__doc__)
    errors = check(root, final=sys.argv[1:] == ["--final"])
    print("\n".join(errors) or f"voice check: {len(snapshot(root))} posts match the baseline")
    sys.exit(1 if errors else 0)
