"""Write search-index.json, the index the Posts and LeetCode search box fetches (SPEC-site-search).

Usage, from the repo root: python3 -I tools/search/build.py .   (after the post builds)
One entry per listed post: u url, t title, s summary, g tags, c section, h headings. Titles, summaries and headings
only, not full text, which would cost 25 times as much. Sorted, with no timestamps, so a rebuild changes nothing.
"""
import html
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("posts_build", HERE.parent / "posts" / "build.py")
posts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(posts)
lc = posts.lc                                     # tools/leetcode/build.py, as the posts build loads it


def plain(text):
    """Text without the markup inline() renders, so results show what the page shows: `code` and \\( math \\)
    keep their contents, and **bold** and [text](url) keep their text."""
    text = re.sub(r"`([^`]+)`|\\\((.+?)\\\)", lambda m: m.group(1) or m.group(2).strip(), text)
    return re.sub(r"\*\*(.+?)\*\*|\[([^\]]+)\]\([^)\s]+\)", lambda m: m.group(1) or m.group(2), text)


assert plain(r"`O(n)` and \( p \), **b** [x](../y/)") == "O(n) and p, b x"


def entry(url, title, summary, tags, section, headings):
    return {"u": url, "t": plain(title), "s": plain(summary), "g": tags, "c": section, "h": [plain(h) for h in headings]}


def sourced(site, section, label):
    """Posts rendered from Markdown sources, as the list pages list them: those whose page exists."""
    out = []
    for src in sorted((HERE.parent / section / "src").glob("*.md")):
        if (Path(site) / section / src.stem / "index.html").exists():
            meta, body = lc.parse(src)
            headings = re.findall(r"^#{2,3} (.+)$", body, re.M)
            out.append(entry(f"/{section}/{src.stem}/", meta["title"], meta["summary"], meta["tags"],
                             f"{label} · {meta['group']}", headings))
    return out


def tags_of(meta):
    # "October 31, 2026 · 8 min read · ranking · evaluation": the parts after the date and length, minus counts.
    return " · ".join(p for p in meta.split(" · ")[1:] if not re.search(r"\d", p))


def longform(site):
    """The hand-written posts: headings from their pages' text column (the side column's "Contents" is not one)."""
    out = []
    for slug, title, summary, meta in posts.LONGFORM:
        prose = (Path(site) / "posts" / slug / "index.html").read_text().split('class="prose"', 1)[1]
        headings = [html.unescape(re.sub(r"<[^>]+>", "", h)) for h in re.findall(r"<h[23][^>]*>(.*?)</h[23]>", prose, re.S)]
        out.append(entry(f"/posts/{slug}/", title, summary, tags_of(meta), "Posts · Long-form", headings))
    return out


def build(site):
    entries = longform(site) + sourced(site, "posts", "Posts") + sourced(site, "leetcode", "LeetCode")
    entries += [entry(url, t, s, tags_of(m), "Posts · MusicGen", []) for url, t, s, m in posts.MUSICGEN]
    entries.sort(key=lambda e: e["u"])
    text = json.dumps(entries, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    (Path(site) / "search-index.json").write_text(text)
    print(f"search index: {len(entries)} entries, {len(text.encode()) / 1000:.1f} KB")


if __name__ == "__main__":
    build(sys.argv[1])
