"""Render the topic posts (one per vault note) into posts/<slug>/, run their Python, and write posts/index.html.

Usage, from the repo root: python3 -I tools/posts/build.py . [slug ...]   (no slugs = every src/*.md)
Same source format and Markdown engine as tools/leetcode/build.py, which this loads; output is committed HTML.
The posts index lists the hand-written long-form posts first (LONGFORM), then the topic groups, then MusicGen.
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

DATE = "November 1, 2026"
GROUPS = ["Machine learning fundamentals", "Probability", "Statistics and experimentation", "LLMs and generative AI",
          "Recommender systems", "ML systems and design", "Coding and data"]
HEAD = (lc.HEAD.replace('<a href="/posts/">Posts</a>', '<a href="/posts/" aria-current="{current}">Posts</a>')
               .replace('<a href="/leetcode/" aria-current="{current}">LeetCode</a>', '<a href="/leetcode/">LeetCode</a>'))
assert HEAD != lc.HEAD and HEAD.count("{current}") == 1, "nav markup in tools/leetcode/build.py changed"

# Hand-written long-form posts (posts/<slug>/index.html), newest first: (slug, title, summary, meta line)
LONGFORM = [
    ("evaluating-ml-systems", "Evaluating ML systems: classification, ranking, search and generative models",
     "A reference that follows business value down to model metrics for four kinds of system, then covers offline tests, online experiments and the ways each step tends to break.",
     "October 31, 2026 · 8 min read · evaluation · experimentation · reference"),
    ("lora-from-scratch", "LoRA from scratch, proven with asserts",
     "Build a tiny GPT, freeze it, add LoRA adapters, fine-tune only those and fuse them back, checking every claim with an assert. Runs on a laptop CPU in seconds.",
     "October 31, 2026 · 7 min read · fine-tuning · LLM · PyTorch"),
    ("text-to-music-2026", "Text-to-music in 2026: models, reward models and DPO",
     "Building text-to-music from parts, with a tensor shape at every step: codec language models from MusicGen to HeartMuLa, scoring from CLAP to the CMI reward model, and DPO derived from its objective.",
     "October 31, 2026 · 16 min read · music generation · alignment · reward models"),
    ("position-bias", "Position bias: why click logs lie and how to correct them",
     "A ranker trained on raw clicks learns position, not relevance. The examination model, a simulation of how far raw CTR misleads, and the corrections: inverse propensity weighting, dual learning, position features and interleaving.",
     "October 31, 2026 · 6 min read · ranking · recommendation · evaluation"),
]
MUSICGEN = [
    ("/musicgen-prune-inference/", "Judging text-to-music: CLAP best-of-N, then and now",
     "How generated music is judged, from FID and CLIP score to FAD, CLAP and learned music judges; why picking the best of N takes by CLAP is legitimate; and four ways to extend it, including pruning weak takes early.",
     "October 24, 2026 · 10 min read · 1 interactive figure · MusicGen · evaluation"),
    ("/musicgen-prune-inference/decoding/", "Inside MusicGen decoding: tokens, guidance and exact pruning",
     "RVQ and the delay pattern, classifier-free guidance, where a decoding step's time goes on a T4, pruning a running batch without changing the survivors, and the design of a best-of-N music service.",
     "October 24, 2026 · 8 min read · 1 interactive figure · MusicGen · inference · systems"),
]


def build(site, slug):
    meta, body = lc.parse(HERE / "src" / f"{slug}.md")
    status = lc.run_code(slug, body)
    content, toc = lc.render(body)
    minutes = max(1, round(len(re.sub(r"<[^>]+>", " ", content).split()) / 230))
    toc_html = "".join(f'<li><a href="#{h}">{lc.inline(t)}</a></li>' for h, t in toc)
    page = HEAD.format(title=html.escape(meta["title"]), summary=html.escape(meta["summary"]),
                       assets=lc.POST_ASSETS, current="true") + f"""<article class="post-body">
<p class="crumb"><a href="/posts/">Posts</a> · {html.escape(meta["group"])}</p>
<header class="post-head">
  <h1>{lc.inline(meta["title"])}</h1>
  <div class="meta">{DATE} · {minutes} min read · {html.escape(meta["tags"])}</div>
</header>
<nav class="toc" aria-label="Contents"><h2>Contents</h2><ol>{toc_html}</ol></nav>
{content}
</article>
""" + lc.FOOT
    out = Path(site) / "posts" / slug / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"{slug}: {minutes} min, {status}")


def card(href, title, summary, meta, level="h2"):
    return (f'<a class="post" href="{href}">\n  <{level}>{html.escape(title)}</{level}>\n'
            f'  <p>{html.escape(summary)}</p>\n  <div class="meta">{html.escape(meta)}</div>\n</a>')


def index(site):
    posts = []
    for src in sorted((HERE / "src").glob("*.md")):
        if (Path(site) / "posts" / src.stem / "index.html").exists():
            posts.append((lc.parse(src)[0], src.stem))
    groups = []
    for group in GROUPS:
        items = sorted((p for p in posts if p[0]["group"] == group), key=lambda p: p[0]["order"])
        if items:
            lis = "\n".join(f'  <li><a href="{slug}/">{lc.inline(m["title"])}</a><span>{lc.inline(m["summary"])}</span></li>'
                            for m, slug in items)
            groups.append(f'<section class="group" id="{lc.slugify(group)}">\n<h2>{html.escape(group)}</h2>\n'
                          f'<ul class="series">\n{lis}\n</ul>\n</section>')
    leetcode_count = len(list((Path(site) / "leetcode").glob("*/index.html")))
    page = HEAD.format(title="Posts", summary="Technical posts on machine learning, statistics, LLMs, recommender systems and ML systems, each with the code and numbers behind it.",
                       assets="", current="page") + f"""<section class="intro">
  <h1>Posts</h1>
  <p>Technical notes on machine learning, each with the code and numbers behind it: {len(LONGFORM)} long-form posts
  first, then {len(posts)} shorter posts by topic.</p>
</section>
<!-- Generated by tools/posts/build.py. Long-form posts are hand-written; the MusicGen posts are deployed by their own repo. -->
{chr(10).join(card(slug + "/", t, s, m) for slug, t, s, m in LONGFORM)}

{chr(10).join(groups)}

<section id="musicgen" class="group">
  <h2>MusicGen</h2>
  <p>Two posts on generating music with MusicGen and choosing the best take, with interactive figures.</p>
{chr(10).join(card(href, t, s, m, "h3") for href, t, s, m in MUSICGEN)}
</section>

<p class="aside">Looking for algorithms? The <a href="/leetcode/">LeetCode patterns</a> series has {leetcode_count} short posts,
one per pattern, with code that runs.</p>
""" + lc.FOOT
    (Path(site) / "posts" / "index.html").write_text(page)
    print(f"posts index: {len(LONGFORM)} long-form + {len(posts)} topic posts")


if __name__ == "__main__":
    site, slugs = sys.argv[1], sys.argv[2:] or [p.stem for p in sorted((HERE / "src").glob("*.md"))]
    for s in slugs:
        build(site, s)
    index(site)
