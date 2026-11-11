"""Render the old book's pages (SPEC-resume-book) into book/<slug>/, run their Python, and write book/index.html.

Usage, from the repo root: python3 -I tools/book/build.py .
Same source format and Markdown engine as tools/posts/build.py. The pages are unlisted: noindex, no canonical,
feed link, JSON-LD or Markdown copy, and the feed and search builds never read this folder (ADR 0003).
"""
import html
import importlib.util
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("lc_build", HERE.parent / "leetcode" / "build.py")
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)
lc.HERE = HERE                                   # sibling links ../<slug>/ resolve against this folder's sources

DATE = "November 11, 2026"
GROUPS = ["Method", "Project stories"]
HEAD = lc.HEAD.replace(' aria-current="{current}"', "")   # the book is in no nav section
assert HEAD != lc.HEAD and "{current}" not in HEAD, "nav markup in tools/leetcode/build.py changed"
NOINDEX = '<meta name="robots" content="noindex">\n'


def build(site, slug):
    meta, body = lc.parse(HERE / "src" / f"{slug}.md")
    status = lc.run_code(f"book-{slug}", body)
    content, toc = lc.render(body)
    minutes = max(1, round(len(re.sub(r"<[^>]+>", " ", content).split()) / 230))
    page = HEAD.format(title=html.escape(meta["title"]), summary=html.escape(meta["summary"]),
                       assets=NOINDEX + lc.FONTS + lc.POST_ASSETS) + lc.article(
                           meta, DATE, minutes, toc, content, f"/book/#{lc.slugify(meta['group'])}") + lc.FOOT
    out = Path(site) / "book" / slug / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"book/{slug}: {minutes} min, {status}")
    return meta


def index(site, pages):
    sections = []
    for group in GROUPS:
        items = sorted((p for p in pages if p[0]["group"] == group), key=lambda p: p[0]["order"])
        if items:
            sections.append((group, lc.slugify(group), [lc.row(m["title"], m["summary"], f"{slug}/") for m, slug in items], ""))
    assert sum(len(s[2]) for s in sections) == len(pages), "a page's group is not in GROUPS"
    listing, chosen = lc.topic_list(sections)
    page = HEAD.format(title="An old book", summary="Notes on preparing a résumé and telling the projects on it.",
                       assets=NOINDEX + lc.FONTS + chosen) + f"""<section class="intro">
  <h1>An old book</h1>
  <p>You found the book on the mantel. Inside: how I prepare the projects on my résumé to be told well, one
  method and one story per project, each with what the numbers do and do not show.</p>
</section>
{listing}
""" + lc.FOOT
    (Path(site) / "book" / "index.html").write_text(page)
    print(f"book index: {len(pages)} pages")


if __name__ == "__main__":
    site = sys.argv[1]
    index(site, [(build(site, src.stem), src.stem) for src in sorted((HERE / "src").glob("*.md"))])
