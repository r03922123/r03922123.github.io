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
     "Build a tiny GPT, freeze it, add LoRA adapters, fine-tune only those and fuse them back, checking most claims with an assert. Runs on a laptop CPU in seconds.",
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
    page = HEAD.format(title=html.escape(meta["title"]), summary=html.escape(meta["summary"]),
                       assets=lc.FONTS + lc.head_tags(f"/posts/{slug}/", (meta["title"], meta["summary"], DATE), markdown=True)
                       + lc.POST_ASSETS, current="true") + lc.article(meta, DATE, minutes, toc, content,
                                                                      f"/posts/#{lc.slugify(meta['group'])}") + lc.FOOT
    out = Path(site) / "posts" / slug / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"{slug}: {minutes} min, {status}")


def index(site):
    posts = []
    for src in sorted((HERE / "src").glob("*.md")):
        if (Path(site) / "posts" / src.stem / "index.html").exists():
            posts.append((lc.parse(src)[0], src.stem))
    # Long-form first: the home's .lofi-latest lists the first three posts here (scripts/check.py).
    sections = [("Long-form", "long-form", [lc.row(t, s, slug + "/", m) for slug, t, s, m in LONGFORM], "")]
    for group in GROUPS:
        items = sorted((p for p in posts if p[0]["group"] == group), key=lambda p: p[0]["order"])
        if items:
            sections.append((group, lc.slugify(group), [lc.row(m["title"], m["summary"], f"{slug}/") for m, slug in items], ""))
    sections.append(("MusicGen", "musicgen", [lc.row(t, s, href, m) for href, t, s, m in MUSICGEN],
                     "<p>Two posts on generating music with MusicGen and choosing the best take, with interactive figures.</p>\n"))
    listing, chosen = lc.topic_list(sections)
    leetcode_count = len(list((Path(site) / "leetcode").glob("*/index.html")))
    page = HEAD.format(title="Posts", summary="Technical posts on machine learning, statistics, LLMs, recommender systems and ML systems, each with the code and numbers behind it.",
                       assets=lc.FONTS + lc.head_tags("/posts/") + chosen + lc.SEARCH, current="page") + f"""<section class="intro">
  <h1>Posts</h1>
  <p>Technical notes on machine learning, each with the code and numbers behind it: {len(LONGFORM)} long-form posts
  first, then {len(posts)} shorter posts by topic.</p>
</section>
<!-- Generated by tools/posts/build.py. Long-form posts are hand-written; the MusicGen posts are deployed by their own repo. -->
{listing}

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
