"""Check that reframing a post (SPEC-post-voice.md) left its evidence alone, against scripts/voice-baseline.json.

Usage, from the repo root:  python3 scripts/voice_check.py [--final]     compare every post with the baseline
                            python3 scripts/voice_check.py --baseline    write the baseline once, before any edit
A post is a topic source (tools/posts/src/<slug>.md) or a long-form page (posts/<slug>/index.html with no source).
Code blocks, tables, headings, front matter and a long-form page's reference list must stay byte-identical. Inline
code, maths and links are compared as sets: a sentence may move or reuse one, but none may be new or vanish. Prose may
change, but every number in it must already be in the post, none may vanish, and it may grow by at most 10%.
--final also fails a post whose prose did not change at all.
Output names posts, rules and numbers, never prose; it goes through check.py's scrubber, since a term can be a number.
"""
import hashlib
import html
import json
import re
import sys
import traceback
from decimal import Decimal
from pathlib import Path

BASELINE = "scripts/voice-baseline.json"
WORDS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                         "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
WORDS.update(twenty="20", thirty="30", forty="40", fifty="50", sixty="60", seventy="70", eighty="80", ninety="90",
             hundred="100", thousand="1000", million="1000000", billion="1000000000")
# shortcut: no sign, and "64,128,256" reads as one number; "half", "twice" and "first" are left to the reviewer
NUMBER = re.compile(r"(?<![\d.])(?:\d+(?:[,_]\d{3})*(?:\.\d+)?|\.\d+)|\b(?:" + "|".join(WORDS) + r")\b", re.I)
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
    """Pull rx's matches out of text: (the set of matches, whitespace-normalized so rewrapping is free; the rest)."""
    found = {" ".join(s.split()) for s in rx.findall(text)}
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
    links = sorted(set(re.findall(r"\]\(([^)\s]+)\)", text)))
    text = re.sub(r"\]\([^)\s]+\)", "]", text)
    return code, tables, inline, sorted(set(display + maths)), [front] + heads + links, text


def shown(markup):
    """Text as a browser shows it: tags dropped, entities decoded."""
    return html.unescape(re.sub(r"<[^>]+>", " ", markup))


def page(text):
    """Split a long-form page's <article>, leaving out reading-shell's own header, nav and aside; any other one is prose.
    Code and tables are compared as markup; their numbers are read as a browser shows them, entities decoded."""
    art = text[text.index("<article"):text.index("</article>")]
    art = re.sub(r'<(header|nav|aside) class="(?:post-head|toc|post-side)".*?</\1>', " ", art, flags=re.S)
    refs, art = re.findall(r'<ol class="refs">.*?</ol>', art, re.S), re.sub(r'<ol class="refs">.*?</ol>', " ", art, flags=re.S)
    code, art = re.findall(r"<pre\b.*?</pre>", art, re.S), re.sub(r"<pre\b.*?</pre>", " ", art, flags=re.S)
    tables, art = re.findall(r"<table\b.*?</table>", art, re.S), re.sub(r"<table\b.*?</table>", " ", art, flags=re.S)
    heads, art = re.findall(r"<h[1-6]\b.*?</h[1-6]>", art, re.S), re.sub(r"<h[1-6]\b.*?</h[1-6]>", " ", art, flags=re.S)
    inline, art = spans(re.compile(r"<code\b.*?</code>", re.S), art)
    links = sorted(set(re.findall(r'\b(?:href|id)="([^"]*)"', art)))
    maths, prose = spans(MATHS, shown(art))
    return code, tables, inline, maths, heads + refs + links, prose


def posts(root):
    srcs = sorted(root.glob("tools/posts/src/*.md"))
    stems = {p.stem for p in srcs}
    longs = sorted(p for p in root.glob("posts/*/index.html") if p.parent.name not in stems)
    return srcs + longs


def snapshot(root):
    snap = {}
    for path in posts(root):
        text = path.read_text()
        md = path.suffix == ".md"
        code, tables, inline, maths, frozen, prose = (markdown if md else page)(text)
        evidence = "\n".join(code + tables + inline + maths)
        snap[path.relative_to(root).as_posix()] = {
            "code": digest(code), "tables": digest(tables), "inline": digest(inline), "maths": digest(maths),
            "frozen": digest(frozen), "prose": digest(prose.split()), "words": len(prose.split()),
            "numbers": sorted(numbers(prose), key=Decimal),
            "evidence": sorted(numbers(evidence if md else shown(evidence)), key=Decimal)}
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


def main(argv):
    from check import DENYLIST, scrubber            # the same scrubber and fail-closed rule as check.py's main
    root = Path(__file__).resolve().parent.parent
    if argv == ["--baseline"]:
        write_baseline(root)
        return 0
    if argv not in ([], ["--final"]):
        sys.exit(__doc__)
    if not DENYLIST.exists():
        sys.exit(f"missing {DENYLIST}: the output could not be scrubbed")
    terms = [t for t in (line.strip() for line in DENYLIST.read_text(encoding="utf-8-sig").splitlines())
             if t and not t.startswith("#")]
    try:
        errors = check(root, final=argv == ["--final"])
    except Exception:                                 # a traceback can quote page text: scrub it too
        errors = traceback.format_exc().splitlines()
    print("\n".join(map(scrubber(terms), errors)) or f"voice check: {len(posts(root))} posts match the baseline")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
