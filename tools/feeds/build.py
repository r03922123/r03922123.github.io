"""Write the site's machine-readable files: rss.xml, sitemap.xml, robots.txt, llms.txt, and an index.md beside each
post that has a Markdown source (SPEC-post-feeds).

Usage, from the repo root: python3 -I tools/feeds/build.py .   (after tools/posts/build.py and tools/leetcode/build.py)
Nothing here reads the clock, so two runs in a row write the same bytes and the rebuild gate stays clean.
"""
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from xml.sax.saxutils import escape

SITE = "https://r03922123.github.io"
NAME = "r03922123"
SUMMARY = ("Technical posts on machine learning, statistics, LLMs, recommender systems and ML systems, and short posts "
           "on LeetCode patterns, each with the code and numbers behind it.")
TAIPEI = timezone(timedelta(hours=8))
TOOLS = Path(__file__).resolve().parent.parent


def load(name, path):
    # Each build on its own: tools/posts/build.py loads a private copy of the LeetCode build and points that copy's
    # HERE at the posts folder, so borrowing that copy would read the posts sources as LeetCode ones.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lc = load("feeds_leetcode", TOOLS / "leetcode" / "build.py")
posts = load("feeds_posts", TOOLS / "posts" / "build.py")


def day(text):
    return datetime.strptime(text, "%B %d, %Y")


def rss_date(text):
    """'November 1, 2026' to RFC 822, at midnight Taipei time; format_datetime ignores the locale, so the bytes never vary."""
    return format_datetime(day(text).replace(tzinfo=TAIPEI))


def sourced(module, section, label=""):
    """(section heading, path, title, summary, date, Markdown) per source, in its listing's order: group, then order."""
    parsed = [(lc.parse(src), src.stem) for src in sorted((module.HERE / "src").glob("*.md"))]   # parse reads no HERE
    out = []
    for group in module.GROUPS:
        for (meta, body), slug in sorted((p for p in parsed if p[0][0]["group"] == group), key=lambda p: p[0][0]["order"]):
            out.append((label + group, f"/{section}/{slug}/", meta["title"], meta["summary"], module.DATE,
                        f"# {meta['title']}\n\n{body}\n"))     # the title stands in for the front matter
    return out


def entries():
    """Every post in listing order: the posts index (long-form, topics, MusicGen), then the LeetCode series."""
    longform = [("Long-form posts", f"/posts/{slug}/", t, s, meta.split(" · ")[0], None) for slug, t, s, meta in posts.LONGFORM]
    music = [("MusicGen", href, t, s, meta.split(" · ")[0], None) for href, t, s, meta in posts.MUSICGEN]
    return longform + sourced(posts, "posts") + music + sourced(lc, "leetcode", "LeetCode patterns: ")


def rss(items):
    newest = sorted(items, key=lambda p: day(p[4]), reverse=True)      # sorted() is stable: ties keep listing order
    body = "".join(f"""  <item>
    <title>{escape(title)}</title>
    <link>{escape(SITE + path)}</link>
    <guid isPermaLink="true">{escape(SITE + path)}</guid>
    <description>{escape(summary)}</description>
    <pubDate>{rss_date(date)}</pubDate>
  </item>
""" for _, path, title, summary, date, _ in newest)
    # No author: RSS wants an email there and says to omit it on a one-author site. No lastBuildDate: it would change every run.
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
<channel>
  <title>{NAME}</title>
  <link>{SITE}/</link>
  <description>{escape(SUMMARY)}</description>
  <language>en</language>
  <atom:link href="{SITE}/rss.xml" rel="self" type="application/rss+xml"/>
{body}</channel>
</rss>
"""


def sitemap(items):
    # loc only: Google ignores priority and changefreq, and there is no accurate per-post lastmod.
    paths = ["/", "/about.html", "/posts/", "/leetcode/"] + [p[1] for p in items]
    urls = "".join(f"  <url><loc>{escape(SITE + path)}</loc></url>\n" for path in paths)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{urls}</urlset>
""", len(paths)


def llms(items):
    sections = {}
    for heading, path, title, summary, _, md in items:            # a post with a Markdown copy links to that copy
        sections.setdefault(heading, []).append(f"- [{title}]({SITE}{path}{'index.md' if md else ''}): {summary}")
    parts = [f"## {heading}\n\n" + "\n".join(lines) for heading, lines in sections.items()]
    return f"# {NAME}\n\n> {SUMMARY}\n\n" + "\n\n".join(parts) + "\n"


def main(site):
    site = Path(site)
    items = entries()
    copies = [(path, md) for _, path, _, _, _, md in items if md]
    for path, md in copies:
        out = site / path.strip("/") / "index.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
    xml, url_count = sitemap(items)
    (site / "rss.xml").write_text(rss(items))
    (site / "sitemap.xml").write_text(xml)
    (site / "robots.txt").write_text(f"User-agent: *\nAllow: /\n\nSitemap: {SITE}/sitemap.xml\n")
    (site / "llms.txt").write_text(llms(items))
    print(f"feeds: rss.xml {len(items)} items, sitemap.xml {url_count} URLs, {len(copies)} index.md files")


if __name__ == "__main__":
    main(sys.argv[1])
