"""Render LeetCode pattern posts from light Markdown into the site template, run their Python, write the series page.

Usage, from the repo root: python3 -I tools/leetcode/build.py . [slug ...]   (no slugs = every src/*.md)
Output is committed HTML under leetcode/; GitHub Pages serves it as is, so the site itself has no build step.
Source format: front matter (title, summary, group, order) between '---' lines, then
## / ### headings, paragraphs, '- ' and '1. ' lists, ```python fences, `code`, **bold**, [text](url), \\( math \\).
"""
import html
import re
import tempfile
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATE = "October 31, 2026"
GROUPS = ["Two pointers, windows and prefix sums", "Sorting, searching and strings", "Stacks, queues, heaps and hashing",
          "Trees and range queries", "Graphs", "Dynamic programming, bits and games"]

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · r03922123</title>
<meta name="description" content="{summary}">
<link rel="stylesheet" href="/style.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.10.0/styles/github.min.css" media="(prefers-color-scheme: light)">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.10.0/styles/github-dark.min.css" media="(prefers-color-scheme: dark)">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.10.0/highlight.min.js"></script>
<script defer src="/post.js"></script>
</head>
<body>
<div class="wrap">
<header>
  <a class="home" href="/">r03922123</a>
  <nav><a href="/">Posts</a><a href="/leetcode/">LeetCode</a><a href="/about.html">About</a><a href="https://github.com/r03922123">GitHub</a></nav>
</header>
<main>
"""
FOOT = """</main>
<footer>Personal site. Not affiliated with any company.</footer>
</div>
</body>
</html>
"""


def inline(text):
    parts = re.split(r"(`[^`]+`|\\\(.+?\\\))", text)   # keep code spans and inline math away from markup
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            out.append(f"<code>{html.escape(p[1:-1], quote=False)}</code>")
        elif p.startswith("\\("):
            out.append(html.escape(p, quote=False))
        else:
            p = html.escape(p, quote=False)
            p = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", p)
            p = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", link, p)
            out.append(p)
    return "".join(out)


def link(m):
    text, href = m.group(1), m.group(2)
    sibling = re.fullmatch(r"\.\./([a-z0-9-]+)/", href)
    if sibling and not (HERE / "src" / f"{sibling.group(1)}.md").exists():
        return text                                  # that post is not written yet; becomes a link on a later build
    return f'<a href="{href}">{text}</a>'


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def render(body):
    out, toc, para, lst, lst_tag = [], [], [], [], None
    lines = body.split("\n")

    def flush():
        nonlocal para, lst, lst_tag
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para = []
        if lst:
            out.append(f"<{lst_tag}>" + "".join(f"<li>{inline(i)}</li>" for i in lst) + f"</{lst_tag}>")
            lst, lst_tag = [], None

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush()
            lang = line[3:].strip() or "plaintext"
            j = i + 1
            while not lines[j].startswith("```"):
                j += 1
            code = html.escape("\n".join(lines[i + 1:j]), quote=False)
            out.append(f'<pre><code class="language-{lang}">{code}</code></pre>')
            i = j + 1
            continue
        if line.startswith("\\["):                    # display math block, kept verbatim until \]
            flush()
            j = i
            while "\\]" not in lines[j]:
                j += 1
            out.append(html.escape("\n".join(lines[i:j + 1]), quote=False))
            i = j + 1
            continue
        if line.startswith("|"):                       # table: header row, separator row, body rows
            flush()
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in lines[i:j]]
            head = "".join(f"<th>{inline(c)}</th>" for c in rows[0])
            body_rows = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows[2:])
            out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body_rows}</tbody></table>")
            i = j
            continue
        m = re.match(r"(#{2,3}) (.+)", line)
        if m:
            flush()
            level, text = len(m.group(1)), m.group(2)
            hid = slugify(text)
            if level == 2:
                toc.append((hid, text))
            out.append(f'<h{level} id="{hid}">{inline(text)}</h{level}>')
        elif re.match(r"- |\d+\. ", line):
            tag = "ul" if line.startswith("- ") else "ol"
            if para or (lst and tag != lst_tag):
                flush()
            lst_tag = tag
            lst.append(re.sub(r"^(- |\d+\. )", "", line))
        elif line.startswith("  ") and lst:             # continuation of a list item
            lst[-1] += " " + line.strip()
        elif not line.strip():
            flush()
        else:
            if lst:
                flush()
            para.append(line.strip())
        i += 1
    flush()
    return "\n".join(out), toc


def parse(path):
    text = path.read_text()
    _, front, body = text.split("---\n", 2)
    meta = dict(line.split(": ", 1) for line in front.strip().split("\n"))
    meta["order"] = int(meta.get("order", 0))
    return meta, body.strip()


def run_code(slug, body):
    blocks = re.findall(r"```python\n(.*?)```", body, re.S)
    if not blocks:
        return "no python"
    script = Path(tempfile.gettempdir()) / f"leetcode-{slug}.py"   # all blocks of a post, run as one script
    script.write_text("\n\n".join(blocks))
    r = subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True, timeout=120)
    if r.returncode:
        sys.exit(f"{slug}: code failed\n{r.stderr[-2000:]}")
    return f"{len(blocks)} python blocks pass"


def build(site, slug):
    meta, body = parse(HERE / "src" / f"{slug}.md")
    status = run_code(slug, body)
    content, toc = render(body)
    words = len(re.sub(r"<[^>]+>", " ", content).split())
    minutes = max(1, round(words / 230))
    toc_html = "".join(f'<li><a href="#{h}">{inline(t)}</a></li>' for h, t in toc)
    page = HEAD.format(title=html.escape(meta["title"]), summary=html.escape(meta["summary"])) + f"""<article class="post-body">
<p class="crumb"><a href="/leetcode/">LeetCode patterns</a> · {html.escape(meta["group"])}</p>
<header class="post-head">
  <h1>{inline(meta["title"])}</h1>
  <div class="meta">{DATE} · {minutes} min read · {html.escape(meta["tags"])}</div>
</header>
<nav class="toc"><h2>Contents</h2><ol>{toc_html}</ol></nav>
{content}
</article>
""" + FOOT
    out = Path(site) / "leetcode" / slug / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"{slug}: {minutes} min, {status}")
    return meta, minutes


def series(site):
    posts = []
    for src in sorted((HERE / "src").glob("*.md")):
        if (Path(site) / "leetcode" / src.stem / "index.html").exists():
            meta, body = parse(src)
            posts.append((meta, src.stem))
    sections = []
    for group in GROUPS:
        items = sorted((p for p in posts if p[0]["group"] == group), key=lambda p: p[0]["order"])
        if not items:
            continue
        lis = "\n".join(f'  <li><a href="{slug}/">{inline(m["title"])}</a><span>{inline(m["summary"])}</span></li>' for m, slug in items)
        sections.append(f'<h2 id="{slugify(group)}">{html.escape(group)}</h2>\n<ul class="series">\n{lis}\n</ul>')
    page = HEAD.format(title="LeetCode patterns", summary="Short posts on the algorithm patterns behind LeetCode problems: the idea, a reusable Python template, worked problems and practice links.") + f"""<section class="intro">
  <h1>LeetCode patterns</h1>
  <p>{len(posts)} short posts, one per pattern: the idea in plain words, a reusable Python template, worked
  problems with their complexity, and problems to practice on. Every code block on these pages runs, and its
  asserts pass.</p>
</section>
{chr(10).join(sections)}
""" + FOOT
    (Path(site) / "leetcode" / "index.html").write_text(page)
    print(f"series page: {len(posts)} posts")


if __name__ == "__main__":
    site, slugs = sys.argv[1], sys.argv[2:] or [p.stem for p in sorted((HERE / "src").glob("*.md"))]
    for s in slugs:
        build(site, s)
    series(site)
