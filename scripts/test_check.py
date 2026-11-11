# Tests for check.py: a clean site passes; each planted fault is reported.
import ast
import importlib.util
import io
import json
import os
import re
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock
from urllib.parse import unquote

from check import check, main, scrubber

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>t</title><meta name="description" content="d"></head>
<body>{body}</body></html>
"""
DENY = ["secret-term", "555-0100"]
# posts/index.html must keep the MusicGen section and its two posts, which another repo deploys
MUSICGEN = ('<section id="musicgen"><h2>MusicGen</h2></section>'
            '<a href="/musicgen-prune-inference/">j</a><a href="/musicgen-prune-inference/decoding/">d</a>')
MUSICGEN_PATHS = ("/musicgen-prune-inference/", "/musicgen-prune-inference/decoding/")
SITE = "https://r03922123.github.io"
RSS = '<?xml version="1.0" encoding="UTF-8"?>\n<rss version="2.0"><channel><title>r</title>{}</channel></rss>\n'
ITEM = f"<item><title>A</title><link>{SITE}/posts/a/</link><pubDate>Sun, 01 Nov 2026 00:00:00 +0800</pubDate></item>"
# The other listed posts of feed_site, so rss.xml and llms.txt are complete (posts/a's item is ITEM).
OTHER_POSTS = ("/posts/c/", *MUSICGEN_PATHS, "/leetcode/b/")
ITEMS = "".join(f"<item><link>{SITE}{u}</link><pubDate>Sun, 01 Nov 2026 00:00:00 +0800</pubDate></item>" for u in OTHER_POSTS)
LLMS = "# r\n\n> s\n\n" + "".join(f"- [x]({SITE}{u}{'index.md' if u in ('/posts/a/', '/leetcode/b/') else ''}): y\n"
                                  for u in ("/posts/a/", *OTHER_POSTS))
SITEMAP = '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{}</urlset>\n'
FIXED = ("/", "/about.html", "/posts/", "/leetcode/")
TAGGED = PAGE.replace("</head>", "{head}</head>")
LD = '<script type="application/ld+json">{"@type": "BlogPosting"}</script>\n'
SHELL_FONTS = ("source-serif-4/source-serif-4-latin-wght-normal.woff2", "inter/inter-latin-wght-normal.woff2")
PRELOAD = '<link rel="preload" href="/fonts/{}" as="font" type="font/woff2" crossorigin>\n'
PRELOADS = "".join(map(PRELOAD.format, SHELL_FONTS))
ARTICLE = '<article class="post-body"><aside class="post-side"></aside><div class="prose"></div></article>'


def head(path, post=False, md=False):
    """The head tags required once the feed build exists: canonical and feed link on every page; on posts, JSON-LD,
    and the Markdown alternate when the post has a source."""
    tags = f'<link rel="canonical" href="{SITE}{path}">\n<link rel="alternate" type="application/rss+xml" href="{SITE}/rss.xml">\n'
    return tags + (f'<link rel="alternate" type="text/markdown" href="{SITE}{path}index.md">\n' if md else "") + (LD if post else "")


class CheckTest(unittest.TestCase):
    def site(self, files):
        root = Path(tempfile.mkdtemp())
        for path, text in files.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            if isinstance(text, bytes):
                (root / path).write_bytes(text)
            else:
                (root / path).write_text(text)
        return root

    def findings(self, files):
        return check(self.site(files), DENY)

    def test_clean_site_passes(self):
        files = {
            "index.html": PAGE.format(body='<a href="posts/a/">a</a> <a href="https://x.org">x</a>'),
            "posts/index.html": PAGE.format(body='<a href="a/">a</a>' + MUSICGEN),
            "posts/a/index.html": PAGE.format(body='<a href="../../">home</a>'),
            "style.css": "body{}",
        }
        self.assertEqual(self.findings(files), [])

    def test_denylist_term_in_any_text_file(self):
        out = self.findings({"index.html": PAGE.format(body=""), "notes.md": "has Secret-Term here"})
        self.assertEqual(len(out), 1)
        self.assertIn("notes.md:1", out[0])

    def test_encoded_newline_does_not_shift_the_line(self):
        self.assertEqual(self.findings({"notes.md": "x%0Ay\nsecret-term\n"}), ["notes.md:2: private term 'secret-term'"])

    def test_every_text_file_is_scanned(self):
        files = {name: "has secret-term here" for name in ("notes.yml", "CNAME", "resume.tex", "data.csv")}
        files["img.png"] = b"\x89PNG\0secret-term"                       # binary files are skipped
        out = self.findings(files)
        self.assertEqual(out, [f"{name}:1: private term 'secret-term'" for name in sorted(files) if name != "img.png"])

    def test_htm_pages_get_page_checks(self):
        out = self.findings({"page.htm": "<html><body>see [[note]]</body></html>"})
        self.assertEqual(len(out), 5)  # vault wikilink, then lang, viewport, title, description

    def test_denylist_matches_whole_words_only(self):
        out = check(self.site({"index.html": PAGE.format(body="Cacmeal\nthen Acme.")}), ["acme"])
        self.assertEqual(out, [f"index.html:{PAGE.count(chr(10)) + 1}: private term 'acme'"])

    def test_multiword_term_split_by_whitespace_entities_or_tags(self):
        text = "worked at Acme\nCorp.\nAcme&nbsp;Corp\n<b>acme</b> corp\nAcme  Corp\nAcmeCorp, Acme Corporation\n"
        out = check(self.site({"notes.md": text}), ["acme corp"])
        self.assertEqual(out, [f"notes.md:{n}: private term 'acme corp'" for n in (1, 3, 4, 5)])

    def test_phone_term_in_national_and_separated_forms(self):
        forms = ["0900-123-456", "0900 123 456", "0900.123.456", "(0900) 123-456", "0900&#8209;123&#8209;456",
                 "+886 (0)900123456", "tel:0900-123-456", "0900123456"]
        out = check(self.site({"notes.md": "\n".join(forms)}), ["900123456"])
        self.assertEqual(out, [f"notes.md:{n}: private term '900123456'" for n in range(1, len(forms) + 1)])

    def test_phone_term_not_matched_inside_longer_numbers(self):
        out = check(self.site({"notes.md": "10900123456\n09001234567"}), ["900123456"])
        self.assertEqual(out, [])

    def test_national_mobile_number_in_html(self):
        out = self.findings({"index.html": PAGE.format(body="call 0911-222-333\nid 109112223334")})
        self.assertEqual(out, [f"index.html:{PAGE.count(chr(10))}: phone number (national)"])

    def test_phone_numbers(self):
        out = self.findings({"index.html": PAGE.format(body="call 555-0100 or +886 905 123 456")})
        self.assertEqual(len(out), 2)

    def test_vault_syntax_in_html(self):
        out = self.findings({"index.html": PAGE.format(body="see [[note]]\n> [!tip] x")})
        self.assertEqual(len(out), 2)

    def test_escaped_callout_in_generated_page(self):
        out = self.findings({"index.html": PAGE.format(body="<p>&gt; [!tip] x</p>")})
        self.assertEqual(out, [f"index.html:{PAGE.count(chr(10))}: vault callout"])

    def test_leetcode_sources_get_page_patterns_but_other_markdown_does_not(self):
        text = "> [!tip] x\nSee [[Heap notes]]\n"
        out = self.findings({"tools/leetcode/src/heap.md": text, "SPEC.md": text})
        self.assertEqual(out, ["tools/leetcode/src/heap.md:1: vault callout", "tools/leetcode/src/heap.md:2: vault wikilink"])

    def test_nested_lists_in_code_are_not_wikilinks(self):
        body = "grid = [[1, 2], [3, 4]]\nmrr([[0, 1, 0], [1, 0, 0]])\nmatrix[[i]]\ndf[[\"col\"]]"
        self.assertEqual(self.findings({"index.html": PAGE.format(body=body)}), [])

    def test_internal_hosts_in_html(self):
        out = self.findings({"index.html": PAGE.format(body="s3://bucket/x\na.confluence.example")})
        self.assertEqual(len(out), 2)

    def test_broken_internal_link(self):
        out = self.findings({"index.html": PAGE.format(body='<a href="missing.html">x</a> <a href="#top">t</a>')})
        self.assertEqual(len(out), 1)
        self.assertIn("missing.html", out[0])

    def test_root_relative_link_resolves_from_repo_root(self):
        files = {"index.html": PAGE.format(body=""), "a/b.html": PAGE.format(body='<a href="/style.css">s</a>'),
                 "style.css": ""}
        self.assertEqual(self.findings(files), [])

    def test_external_project_repo_paths_are_not_checked(self):
        out = self.findings({"index.html": PAGE.format(body='<a href="/musicgen-prune-inference/">p</a>')})
        self.assertEqual(out, [])

    def test_typo_in_top_level_folder_is_a_broken_link(self):
        files = {"index.html": PAGE.format(body='<a href="/leetcod/">l</a> <a href="/postz/">p</a>'),
                 "leetcode/index.html": PAGE.format(body="")}
        self.assertEqual(self.findings(files), ["index.html: broken link /leetcod/", "index.html: broken link /postz/"])

    def test_link_case_must_match_exactly(self):
        files = {"index.html": PAGE.format(body='<a href="/About.html">a</a> <a href="Posts/">p</a>'),
                 "about.html": PAGE.format(body=""), "posts/index.html": PAGE.format(body=MUSICGEN)}
        self.assertEqual(self.findings(files), ["index.html: broken link /About.html", "index.html: broken link Posts/"])

    def test_single_quoted_links_are_checked(self):
        out = self.findings({"index.html": PAGE.format(body="<a href='missing.html'>x</a>")})
        self.assertEqual(out, ["index.html: broken link missing.html"])

    def test_dot_segments_resolve(self):
        files = {"index.html": PAGE.format(body='<a href="./a/../style.css">s</a> <a href="a/../../x.css">x</a>'),
                 "style.css": "", "x.css": ""}
        self.assertEqual(self.findings(files), ["index.html: broken link a/../../x.css"])

    # SPEC-reading-shell: a contents link or a room door with no matching id lands at the top of the page, silently.
    def fragment_site(self, body):
        return {"index.html": PAGE.format(body=body), "about.html": PAGE.format(body='<h2 id="tiktok">t</h2>'),
                "posts/index.html": PAGE.format(body='<h2 id="intro">i</h2>' + MUSICGEN)}

    def test_matching_fragments_pass(self):
        body = ('<h2 id="here">h</h2><a href="#here">a</a><a href="about.html#tiktok">b</a><a href="/posts/#intro">c</a>'
                '<a href="#">d</a><a href="#top">e</a><a href="/musicgen-prune-inference/#x">f</a>')
        self.assertEqual(self.findings(self.fragment_site(body)), [])

    def test_missing_fragment_on_another_page(self):
        out = self.findings(self.fragment_site('<a href="about.html#missing">x</a>'))
        self.assertEqual(out, ["index.html: broken fragment about.html#missing"])

    def test_missing_fragment_on_a_folder_page(self):
        out = self.findings(self.fragment_site('<a href="/posts/#missing">x</a>'))
        self.assertEqual(out, ["index.html: broken fragment /posts/#missing"])

    def test_missing_fragment_on_the_same_page(self):
        out = self.findings(self.fragment_site('<h2 id="here">h</h2><a href="#missing">x</a>'))
        self.assertEqual(out, ["index.html: broken fragment #missing"])

    def test_missing_page_with_a_fragment_is_reported_once(self):
        out = self.findings(self.fragment_site('<a href="nope.html#x">x</a>'))
        self.assertEqual(out, ["index.html: broken link nope.html#x"])

    def test_fragment_must_match_the_id_case(self):
        out = self.findings(self.fragment_site('<h2 id="Here">h</h2><a href="#here">x</a>'))
        self.assertEqual(out, ["index.html: broken fragment #here"])

    def test_top_fragment_ignores_case(self):
        # Browsers scroll to the top for #top in any case when no element has that id.
        self.assertEqual(self.findings(self.fragment_site('<a href="#TOP">x</a><a href="#Top">y</a>')), [])

    def test_fragment_is_percent_decoded(self):
        self.assertEqual(self.findings(self.fragment_site('<h2 id="a b">h</h2><a href="#a%20b">x</a>')), [])

    def test_fragment_on_a_non_html_file_is_not_checked(self):
        site = {**self.fragment_site('<a href="/notes.txt#x">x</a>'), "notes.txt": "n"}
        self.assertEqual(self.findings(site), [])

    # SPEC-reading-shell: each page on the stylesheet but the home preloads both reading fonts, and each article has
    # the side and text columns, so a hand-written page cannot keep the old markup unnoticed.
    def shell_site(self, about_head, article):
        css = '<link rel="stylesheet" href="/style.css">\n'
        return {**{f"fonts/{font}": b"\0" for font in SHELL_FONTS}, "style.css": "",
                "index.html": TAGGED.format(head=css, body=""), "about.html": TAGGED.format(head=css + about_head, body=""),
                "posts/index.html": PAGE.format(body='<a href="a/">a</a>' + MUSICGEN), "posts/a/index.html": PAGE.format(body=article)}

    def test_shell_site_passes(self):
        self.assertEqual(self.findings(self.shell_site(PRELOADS, ARTICLE)), [])

    def test_reading_page_without_a_font_preload(self):
        out = self.findings(self.shell_site(PRELOAD.format(SHELL_FONTS[0]), ARTICLE))
        self.assertEqual(out, [f"about.html: no preload for /fonts/{SHELL_FONTS[1]}"])

    def test_article_without_the_side_column(self):
        out = self.findings(self.shell_site(PRELOADS, ARTICLE.replace('<aside class="post-side"></aside>', "")))
        self.assertEqual(out, ["posts/a/index.html: article without post-side"])

    def test_article_without_the_prose_column(self):
        out = self.findings(self.shell_site(PRELOADS, ARTICLE.replace('class="prose"', 'class="text"')))
        self.assertEqual(out, ["posts/a/index.html: article without prose"])

    def test_missing_head_fields(self):
        out = self.findings({"index.html": "<html><body>x</body></html>"})
        self.assertEqual(len(out), 4)  # lang, viewport, title, description

    def test_unpinned_cdn_script(self):
        body = ('<script src="https://cdn.jsdelivr.net/npm/katex/dist/katex.min.js"></script>'
                '<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>')
        out = self.findings({"index.html": PAGE.format(body=body)})
        self.assertEqual(len(out), 1)
        self.assertIn("unpinned", out[0])

    def test_leetcode_post_must_be_listed_on_series_page(self):
        files = {"index.html": PAGE.format(body='<a href="leetcode/">series</a>'),
                 "leetcode/index.html": PAGE.format(body='<a href="two-pointers/">a</a>'),
                 "leetcode/two-pointers/index.html": PAGE.format(body=""),
                 "leetcode/heap/index.html": PAGE.format(body="")}
        out = self.findings(files)
        self.assertEqual(out, ["leetcode/index.html: leetcode/heap/ is not listed"])

    def test_svg_links_need_an_aria_label(self):
        body = '<svg viewBox="0 0 10 10"><a href="#x" aria-label="X"><rect/></a><a href="#y"><rect/></a></svg>' \
               '<h2 id="x">x</h2><h2 id="y">y</h2>'                               # the doors' targets exist
        out = self.findings({"index.html": PAGE.format(body=body)})
        self.assertEqual(out, ["index.html: link inside SVG without aria-label: #y"])

    def test_home_weight_budget(self):
        out = self.findings({"index.html": PAGE.format(body="x" * 150_001)})
        self.assertEqual(len(out), 1)
        self.assertIn("budget", out[0])

    def test_post_must_be_listed_on_posts_index_not_just_home(self):
        files = {"index.html": PAGE.format(body='<a href="posts/a/">a</a>'),
                 "posts/index.html": PAGE.format(body=MUSICGEN),
                 "posts/a/index.html": PAGE.format(body="")}
        self.assertEqual(self.findings(files), ["posts/index.html: posts/a/ is not listed"])

    def test_listing_needs_a_real_link(self):
        for body in ('<!-- <a href="a/">a</a> -->', "see posts/a/ soon", '<p title="a/">a</p>'):
            files = {"posts/index.html": PAGE.format(body=body + '<a href="/posts/b/">b</a>' + MUSICGEN),
                     "posts/a/index.html": PAGE.format(body=""), "posts/b/index.html": PAGE.format(body="")}
            self.assertEqual(self.findings(files), ["posts/index.html: posts/a/ is not listed"], body)

    def test_posts_index_needs_the_musicgen_section(self):
        files = {"posts/index.html": PAGE.format(body=MUSICGEN.replace('id="musicgen"', 'id="music"'))}
        self.assertEqual(self.findings(files), ["posts/index.html: missing the #musicgen section"])

    def test_musicgen_section_links_both_posts(self):
        body = '<a href="/musicgen-prune-inference/decoding/">d</a><section id="musicgen"></section>' \
               '<a href="/musicgen-prune-inference/">j</a>'                       # the decoding link comes too early
        out = self.findings({"posts/index.html": PAGE.format(body=body)})
        self.assertEqual(out, ["posts/index.html: #musicgen does not link /musicgen-prune-inference/decoding/"])

    def test_leetcode_counts_match_the_series(self):
        files = {"index.html": PAGE.format(body='<a href="leetcode/" aria-label="LeetCode patterns: 2 posts">n</a>\n'
                                                'LEETCODE · 3\n<li>3 pattern notes</li>'),
                 "posts/index.html": PAGE.format(body=MUSICGEN + "\nThe series has 3 short posts"),
                 "leetcode/index.html": PAGE.format(body='<a href="a/">a</a><a href="b/">b</a>'),
                 "leetcode/a/index.html": PAGE.format(body=""), "leetcode/b/index.html": PAGE.format(body="")}
        body = PAGE.count(chr(10))
        self.assertEqual(self.findings(files), [f"index.html:{body + 1}: says 3 LeetCode posts, leetcode/ has 2",
                                                f"index.html:{body + 2}: says 3 LeetCode posts, leetcode/ has 2",
                                                f"posts/index.html:{body + 1}: says 3 LeetCode posts, leetcode/ has 2"])

    def test_lofi_latest_is_the_top_three_posts(self):
        listing = '<a class="post" href="d/">d</a><a class="post" href="/posts/c/">c</a>' \
                  '<a class="post" href="b/index.html">b</a><a class="post" href="a/">a</a>' + MUSICGEN
        files = {"posts/index.html": PAGE.format(body=listing)}
        files.update({f"posts/{slug}/index.html": PAGE.format(body="") for slug in "abcd"})
        latest = '<ul class="lofi-latest"><li><a href="posts/d/">d</a></li><li><a href="{}">x</a></li>' \
                 '<li><a href="posts/b/">b</a></li></ul><a href="posts/a/">all</a>'
        files["index.html"] = PAGE.format(body=latest.format("posts/c/"))
        self.assertEqual(self.findings(files), [])
        files["index.html"] = PAGE.format(body=latest.format("posts/a/"))
        self.assertEqual(self.findings(files), ["index.html: .lofi-latest lists /posts/d/, /posts/a/, /posts/b/; "
                                                "the posts index starts /posts/d/, /posts/c/, /posts/b/"])

    def test_post_not_listed_on_home(self):
        out = self.findings({"index.html": PAGE.format(body=""), "posts/a/index.html": PAGE.format(body="")})
        self.assertEqual(len(out), 1)
        self.assertIn("posts/a/", out[0])

    def topics_findings(self, cards, rows):
        """SPEC-posts-index: a list page with topic cards (id, the count it says) and sections (id, rows)."""
        nav = "".join(f'<a href="#{sid}"><b>{sid}</b><span>{said}</span></a>' for sid, said in cards)
        groups = "".join(f'<section class="group" id="{sid}"><h2>{sid}</h2><a class="all" href="#topics">All topics</a>'
                         f'<ul class="series">{"<li>row</li>" * n}</ul></section>' for sid, n in rows)
        body = f'<nav class="topics" id="topics">{nav}</nav><div class="groups">{groups}</div>'
        return self.findings({"leetcode/index.html": PAGE.format(body=body)})

    def test_every_section_needs_a_topic_card(self):
        self.assertEqual(self.topics_findings([("a", "2 posts"), ("b", "1 post")], [("a", 2), ("b", 1)]), [])
        self.assertEqual(self.topics_findings([("a", "2 posts")], [("a", 2), ("b", 1)]),
                         ["leetcode/index.html: section #b has no topic card"])

    def test_card_count_must_match_its_rows(self):
        self.assertEqual(self.topics_findings([("a", "3 posts")], [("a", 2)]),
                         ["leetcode/index.html: card #a says 3 posts, its section has 2 rows"])

    def test_card_count_is_read_from_its_count_only(self):
        # A topic name that ends in a number of posts must not stand in for a missing count.
        body = ('<nav class="topics"><a href="#a"><b>Top 3 posts</b><span></span></a></nav>'
                '<section class="group" id="a"><ul><li>1</li><li>2</li><li>3</li></ul></section>')
        self.assertEqual(self.findings({"leetcode/index.html": PAGE.format(body=body)}),
                         ["leetcode/index.html: card #a says no count, its section has 3 rows"])

    def test_listed_post_missing_from_search_index(self):
        # SPEC-site-search: once the index build exists, every post a listing links must be in the index.
        files = {"posts/index.html": PAGE.format(body='<a href="a/">a</a><a href="b/">b</a>' + MUSICGEN),
                 "posts/a/index.html": PAGE.format(body=""), "posts/b/index.html": PAGE.format(body=""),
                 "tools/search/build.py": ""}
        entries = [{"u": u} for u in ("/posts/a/", "/posts/b/", *MUSICGEN_PATHS)]
        self.assertEqual(self.findings({**files, "search-index.json": json.dumps(entries)}), [])
        self.assertEqual(self.findings({**files, "search-index.json": json.dumps(entries[1:])}),
                         ["search-index.json: missing /posts/a/"])
        self.assertEqual(self.findings(files), ["search-index.json: unreadable (FileNotFoundError)"])

    def feed_site(self, drop_url=None, item=ITEM, edit=None, **files):
        """A small site with every feed file and head tag in order: a topic post and a LeetCode post, each with a
        source and its index.md, and a hand-written post c with no source. A keyword replaces one file, None deletes
        it; edit=(path, old, new) changes one page."""
        urls = [u for u in (*FIXED, "/posts/a/", "/posts/c/", *MUSICGEN_PATHS, "/leetcode/b/") if u != drop_url]
        site = {"index.html": TAGGED.format(head=head("/"), body=""),
                "posts/index.html": TAGGED.format(head=head("/posts/"), body='<a href="a/">a</a><a href="c/">c</a>' + MUSICGEN),
                "posts/a/index.html": TAGGED.format(head=head("/posts/a/", post=True, md=True), body=""),
                "posts/c/index.html": TAGGED.format(head=head("/posts/c/", post=True), body=""),
                "posts/a/index.md": "# A\n\nbody\n", "tools/posts/src/a.md": "---\ntitle: A\n---\nbody\n", "tools/feeds/build.py": "",
                "leetcode/index.html": TAGGED.format(head=head("/leetcode/"), body='<a href="b/">b</a>'),
                "leetcode/b/index.html": TAGGED.format(head=head("/leetcode/b/", post=True, md=True), body=""),
                "leetcode/b/index.md": "# B\n\nsee [A](../../posts/a/)\n",
                "tools/leetcode/src/b.md": "---\ntitle: B\n---\nsee [A](../../posts/a/)\n",
                "rss.xml": RSS.format(item + ITEMS), "sitemap.xml": SITEMAP.format("".join(f"<url><loc>{SITE}{u}</loc></url>" for u in urls)),
                "robots.txt": f"User-agent: *\nAllow: /\n\nSitemap: {SITE}/sitemap.xml\n", "llms.txt": LLMS}
        site.update(files)
        if edit:
            path, old, new = edit
            self.assertIn(old, site[path])
            site[path] = site[path].replace(old, new)
        return self.findings({path: text for path, text in site.items() if text is not None})

    def test_feed_site_passes(self):
        self.assertEqual(self.feed_site(), [])

    def test_feed_files_are_required_once_the_feed_build_exists(self):
        # Deleting a feed must fail, not pass; a site without the feed build has nothing to require yet.
        self.assertEqual(self.feed_site(**{"rss.xml": None}), ["rss.xml: missing"])
        self.assertEqual(self.feed_site(**{name: None for name in ("tools/feeds/build.py", "rss.xml", "sitemap.xml",
                                                                    "robots.txt", "llms.txt")}), [])

    def test_listed_post_missing_from_sitemap(self):
        for url in ("/posts/a/", MUSICGEN_PATHS[1], "/leetcode/b/"):
            self.assertEqual(self.feed_site(drop_url=url), [f"sitemap.xml: missing {SITE}{url}"])

    def test_fixed_page_missing_from_sitemap(self):
        # No listing links About, so only the fixed list keeps it in the sitemap.
        self.assertEqual(self.feed_site(drop_url="/about.html"), [f"sitemap.xml: missing {SITE}/about.html"])

    def test_malformed_feed_xml(self):
        out = self.feed_site(**{"rss.xml": "<rss><channel></rss>"})
        self.assertEqual(len(out), 1, out)
        self.assertTrue(out[0].startswith("rss.xml: malformed XML"), out)

    def test_feed_item_without_pubdate(self):
        self.assertEqual(self.feed_site(item=re.sub(r"<pubDate>.*</pubDate>", "", ITEM)), ["rss.xml: item 1 has no pubDate"])

    def test_email_in_feed(self):
        out = self.feed_site(**{"llms.txt": LLMS + "katex@0.16.11\nwrite to someone@example.com\n"})
        self.assertEqual(out, [f"llms.txt:{LLMS.count(chr(10)) + 2}: email address"])

    def test_listed_post_missing_from_rss_or_llms(self):
        self.assertEqual(self.feed_site(item=""), [f"rss.xml: missing {SITE}/posts/a/"])
        out = self.feed_site(**{"llms.txt": LLMS.replace(f"({SITE}/leetcode/b/index.md)", "(/leetcode/b/index.md)")})
        self.assertEqual(out, [f"llms.txt: missing {SITE}/leetcode/b/", "llms.txt: link /leetcode/b/index.md is not on " + SITE])

    def test_feed_link_on_another_host(self):
        out = self.feed_site(item=ITEM.replace(f"{SITE}/posts/a/", "https://example.com/posts/a/"))
        self.assertEqual(out, [f"rss.xml: missing {SITE}/posts/a/", f"rss.xml: link https://example.com/posts/a/ is not on {SITE}"])

    def test_markdown_copy_without_a_source(self):
        # A renamed source leaves its old index.md online unless something notices.
        out = self.feed_site(**{"posts/old/index.md": "# Old\n"})
        self.assertEqual(out, ["posts/old/index.md: no source tools/posts/src/old.md"])

    def test_post_source_without_markdown(self):
        self.assertEqual(self.feed_site(**{"posts/a/index.md": None}), ["posts/a/index.md: missing (source tools/posts/src/a.md)"])
        self.assertEqual(self.feed_site(**{"leetcode/b/index.md": None}),
                         ["leetcode/b/index.md: missing (source tools/leetcode/src/b.md)"])

    def test_broken_link_in_markdown_copy(self):
        # The page drops a link to an unwritten post, but its index.md keeps it live; code is never a link.
        md = "# A\n\nsee [B](../b/), `f[i](x)`, \\( f[i](x) \\)\n\n```python\nops[op](x)\n```\n"
        self.assertEqual(self.feed_site(**{"posts/a/index.md": md}), ["posts/a/index.md: broken link ../b/"])

    def test_markdown_keeps_front_matter(self):
        out = self.feed_site(**{"posts/a/index.md": "---\ntitle: A\n---\nbody\n"})
        self.assertEqual(out, ["posts/a/index.md: front matter kept"])

    def test_canonical_link_must_match_the_page_url(self):
        out = self.feed_site(edit=("posts/a/index.html", f'canonical" href="{SITE}/posts/a/"', f'canonical" href="{SITE}/posts/c/"'))
        self.assertEqual(out, [f"posts/a/index.html: canonical link {SITE}/posts/c/ is not {SITE}/posts/a/"])

    def test_page_needs_exactly_one_canonical_link(self):
        own = f'<link rel="canonical" href="{SITE}/posts/a/">\n'
        out = self.feed_site(edit=("posts/a/index.html", own, own * 2))
        self.assertEqual(out, [f"posts/a/index.html: canonical link {SITE}/posts/a/, {SITE}/posts/a/ is not {SITE}/posts/a/"])

    def test_page_without_canonical_link(self):
        out = self.feed_site(edit=("leetcode/index.html", f'<link rel="canonical" href="{SITE}/leetcode/">\n', ""))
        self.assertEqual(out, ["leetcode/index.html: no canonical link"])

    def test_page_without_feed_link(self):
        out = self.feed_site(edit=("index.html", f'<link rel="alternate" type="application/rss+xml" href="{SITE}/rss.xml">\n', ""))
        self.assertEqual(out, ["index.html: no feed link"])

    def test_post_without_json_ld(self):
        self.assertEqual(self.feed_site(edit=("posts/c/index.html", LD, "")), ["posts/c/index.html: no JSON-LD"])

    def test_json_ld_that_does_not_parse(self):
        out = self.feed_site(edit=("leetcode/b/index.html", '"BlogPosting"}', '"BlogPosting",}'))
        self.assertEqual(len(out), 1, out)
        self.assertTrue(out[0].startswith("leetcode/b/index.html: JSON-LD does not parse"), out)

    def test_email_in_json_ld(self):
        out = self.feed_site(edit=("posts/a/index.html", '"BlogPosting"}', '"BlogPosting", "email": "someone@example.com"}'))
        self.assertEqual(out, ["posts/a/index.html: email address in JSON-LD"])

    def test_sourced_post_without_markdown_alternate(self):
        # posts/c has no source, so it passes without one (see test_feed_site_passes).
        out = self.feed_site(edit=("leetcode/b/index.html", f'type="text/markdown" href="{SITE}/leetcode/b/index.md"',
                                   f'type="text/markdown" href="{SITE}/leetcode/a/index.md"'))
        self.assertEqual(out, ["leetcode/b/index.html: no Markdown alternate"])

    def test_main_block_comes_last(self):
        # unittest.main() exits, so any test class defined after it would not run when this file is executed
        last = ast.parse(Path(__file__).read_text()).body[-1]
        self.assertIn("__main__", ast.unparse(last.test) if isinstance(last, ast.If) else "")


class BuildTest(unittest.TestCase):
    """The page template in tools/leetcode/build.py, which both post builds use."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("lc_build", Path(__file__).resolve().parent.parent / "tools/leetcode/build.py")
        cls.lc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.lc)

    def test_json_ld_has_no_angle_bracket(self):
        # "<!--" or "</script" inside the script would change how the page parses; \u003c is the same JSON.
        title = "a <!-- b </script> <c>"
        tags = self.lc.head_tags("/posts/a/", (title, "s", "November 1, 2026"))
        ld = tags.split('<script type="application/ld+json">', 1)[1].rsplit("</script>", 1)[0]
        self.assertNotIn("<", ld)
        self.assertEqual(json.loads(ld)["headline"], title)

    def test_chosen_card_rule_works_for_an_id_that_starts_with_a_digit(self):
        # #1-a is not a valid CSS selector, so the whole rule would be dropped; an attribute selector is.
        _, chosen = self.lc.topic_list([("1 A", "1-a", ["<li></li>"], "")])
        self.assertIn(':root:has([id="1-a"]:target) .topics a[href="#1-a"]', chosen)


STYLE = Path(__file__).resolve().parent.parent / "style.css"


def tokens(block):
    return dict(re.findall(r"--([a-z-]+):\s*(#[0-9a-fA-F]{6})", block))


def luminance(hex_color):
    channels = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a, b):
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


# The four token blocks (SPEC-blog-theme): reading pages at :root, the common room under body.home, each in light and
# dark. Each block ends at its own closing brace, so a rule after it inside the same @media is still scanned.
PALETTES = {
    "reading paper": r"^:root \{",
    "reading night": r"^@media \(prefers-color-scheme: dark\) \{\s*:root \{",
    "room paper": r"^body\.home \{",
    "room night": r"^@media \(prefers-color-scheme: dark\) \{\s*body\.home \{",
}

# Today's parchment and candlelit values, copied from main's style.css before the move. The room's art and lo-fi page
# were drawn for these, so they are pinned literally: any edit to them must also edit this test.
ROOM = {
    "room paper": {"bg": "#f3e9d2", "fg": "#2b2118", "muted": "#6b5a45", "line": "#dccfb0", "card": "#f8f1df",
                   "accent": "#8a2a1a", "link": "#7a2414", "signal": "#8a6d1f", "gold": "#b8860b",
                   "code-builtin": "#8a4b08", "code-keyword": "#a3172f"},
    "room night": {"bg": "#1a1410", "fg": "#efe3c8", "muted": "#b9a88a", "line": "#3a2e24", "card": "#231b15",
                   "accent": "#e0a85a", "link": "#e8bf7a", "signal": "#f0c869", "gold": "#d9a441",
                   "code-builtin": "#ffa657", "code-keyword": "#ff7b72"},
}
ROOM_FONTS = {   # the home requests no new font: it keeps the system body stack and Cinzel
    "font-body": '-apple-system, BlinkMacSystemFont, "Segoe UI", "Hiragino Sans", Helvetica, Arial, sans-serif',
    "font-display": '"Cinzel", "Iowan Old Style", "Palatino Linotype", Georgia, serif',
}


def palette_spans(css):
    """{name: (start, end)} of each token block, end just past the block's own closing brace."""
    spans = {}
    for name, pattern in PALETTES.items():
        found = list(re.finditer(pattern, css, re.M))
        if len(found) != 1:                       # not assert: python -O would skip it and read the first match
            raise AssertionError(f"{name}: expected one token block, found {len(found)}")
        spans[name] = (found[0].start(), css.index("}", found[0].end()) + 1)
    return spans


def stray_colours(css):
    """Colour literals on every line outside the four token blocks, up to the scene art, which may use literals."""
    rest, last = "", 0
    for a, b in sorted(palette_spans(css).values()):
        rest, last = rest + css[last:a], b
    rest += css[last:].split("/* Scene", 1)[0]
    return re.findall(r"#[0-9a-fA-F]{3,8}\b|rgba?\(", rest)


class ThemeTest(unittest.TestCase):
    """SPEC-theme and SPEC-blog-theme: text tokens meet WCAG AA on every background, in all four palettes; the room
    keeps today's values; colours live in tokens."""

    def variants(self):
        css = STYLE.read_text()
        return {name: tokens(css[a:b]) for name, (a, b) in palette_spans(css).items()}

    def test_text_tokens_meet_aa(self):
        variants = self.variants()
        self.assertEqual(sorted(variants), sorted(PALETTES))
        for name, t in variants.items():
            for text in ("fg", "muted", "link", "accent", "code-builtin", "code-keyword"):
                for surface in ("bg", "card"):
                    ratio = contrast(t[text], t[surface])
                    self.assertGreaterEqual(ratio, 4.5, f"{name}: --{text} on --{surface} is {ratio:.2f}:1")

    def test_room_keeps_todays_colours(self):
        variants = self.variants()
        for name, pinned in ROOM.items():
            self.assertEqual(variants[name], pinned, f"{name} differs from the parchment values on main")

    def test_room_requests_no_new_font(self):
        css = STYLE.read_text()
        a, b = palette_spans(css)["room paper"]
        fonts = dict(re.findall(r"--(font-body|font-display):\s*([^;]+);", css[a:b]))
        self.assertEqual(fonts, ROOM_FONTS)

    def test_no_hard_coded_colours_outside_tokens_and_scene(self):
        self.assertEqual(stray_colours(STYLE.read_text()), [], "colours outside the token blocks")

    def test_stray_colour_inside_a_dark_media_rule_is_found(self):
        # A dark-only tweak placed after the tokens, inside the same @media rule, is outside the token block.
        css = STYLE.read_text()
        for block, colour in ((r":root", "#ff00ff"), (r"body\.home", "#00ff00")):
            css = re.sub(rf"(@media \(prefers-color-scheme: dark\) \{{\s*{block} \{{[^}}]*\}})",
                         rf"\1\n  .stray {{ color: {colour}; }}", css, count=1)
        self.assertEqual(stray_colours(css), ["#ff00ff", "#00ff00"])


LEAK = re.compile(r"acme(?:[\W_]|%20)*corp", re.I)        # the term below in any case, spacing or path form


class MainTest(unittest.TestCase):
    """main() is what CI runs and Actions logs are public: no output line may show a term, the run must still fail,
    and an unset or empty secret must fail closed. The runner masks secret lines, so a clean CI log proves nothing."""

    site = CheckTest.site

    def run_main(self, files, terms="acme corp\n", *args):
        denylist = Path(tempfile.mkdtemp()) / "denylist.local.txt"     # outside the site, as in the repo
        denylist.write_text(terms)
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(out), self.assertRaises(SystemExit) as exit_:
            main(list(args), root=self.site(files), denylist=denylist)
        code = exit_.exception.code
        if isinstance(code, str):                                      # sys.exit("message") prints to stderr
            out.write(code)
        return code not in (0, None), out.getvalue()

    def assert_fails_without_leak(self, files):
        failed, out = self.run_main(files)
        self.assertTrue(failed, out)
        self.assertNotRegex(unquote(out), LEAK)                         # %-encoded forms count as a leak too
        return out

    def test_term_in_a_case_variant_is_found_and_hidden(self):
        out = self.assert_fails_without_leak({"index.html": PAGE.format(body=""), "notes.md": "Worked at ACME Corp."})
        self.assertIn("notes.md:1: private term", out)

    def test_term_in_a_broken_link_is_hidden(self):
        out = self.assert_fails_without_leak({"index.html": PAGE.format(body='<a href="acme-corp.html">x</a>')})
        self.assertIn("broken link", out)

    def test_term_in_a_hyphenated_path_is_found_and_hidden(self):
        out = self.assert_fails_without_leak({"index.html": PAGE.format(body=""), "posts/index.html": PAGE.format(
            body=MUSICGEN), "posts/acme-corp/index.html": PAGE.format(body="")})
        self.assertIn("private term in file name", out)

    def test_term_in_a_binary_file_name_is_found(self):
        for name in ("img/acme_corp.png", "files/Acme%20Corp.pdf", "img/acme.corp.png", "img/%61cme-corp.png"):
            out = check(self.site({name: b"\x89PNG\0"}), ["acme corp"])
            self.assertEqual(out, [f"{name}: private term in file name 'acme corp'"])

    def test_term_joined_by_punctuation_in_text_is_found(self):
        # Contents get the same word gaps as file names: a folder named acme-corp quoted in a page is the term.
        for text in ("x acme-corp y", "x acme_corp y", "x acme.corp y", "x acme/corp y"):
            self.assertEqual(check(self.site({"notes.md": text}), ["acme corp"]), ["notes.md:1: private term 'acme corp'"])

    def test_percent_encoded_term_in_a_link_is_found_and_hidden(self):
        out = self.assert_fails_without_leak({"index.html": PAGE.format(body='<a href="/%61cme%20corp/">x</a>')})
        self.assertIn("private term", out)

    def test_term_inside_a_longer_term_leaves_no_part_showing(self):
        failed, out = self.run_main({"notes.md": "Acme Corp"}, "acme\nacme corp\n")
        self.assertTrue(failed)
        self.assertNotIn("corp", out.lower())

    def test_term_inside_another_word_is_left_alone(self):
        # A label inside "planning" would mark "ann" as private next to public text that fills the gap.
        scrub = scrubber(["ann", "acme corp"])
        self.assertEqual(scrub("posts/planning.html: channel acmecorporate"), "posts/planning.html: channel acmecorporate")
        self.assertEqual(scrub("img/acmecorp.png Ann"), "img/[term #2].png [term #1]")

    def test_phone_term_in_a_path_is_hidden(self):
        # Detection reads _ and / in a name as gaps, so the scrubber must hide a number written with them.
        scrub = scrubber(["0912 345 678"])
        for line in ("img/tel0912_345_678.png", "img/0912/345/678x.png", "img/886_912_345_678.png"):
            self.assertNotIn("345", scrub(line), line)

    def test_term_with_punctuation_in_a_file_name_is_found(self):
        out = check(self.site({"posts/acme-corp/index.html": "x"}), ["acme-corp"], terms_only=True)
        self.assertEqual(out, ["posts/acme-corp/index.html: private term in file name 'acme-corp'"])

    def test_term_with_punctuation_is_found_in_its_other_spellings(self):
        # Names and pages are read with - _ . / as gaps, so the listed term must be read that way too.
        for name in ("img/acme_corp.png", "img/acme.corp.png", "files/acme%20corp.pdf"):
            self.assertEqual(check(self.site({name: b"\x89PNG\0"}), ["acme-corp"]),
                             [f"{name}: private term in file name 'acme-corp'"], name)
        self.assertEqual(check(self.site({"img/acme-io.png": b"\x89PNG\0"}), ["acme.io"]),
                         ["img/acme-io.png: private term in file name 'acme.io'"])
        self.assertEqual(check(self.site({"notes.md": "x acme corp y"}), ["acme-corp"]),
                         ["notes.md:1: private term 'acme-corp'"])

    def test_raw_form_is_scrubbed_before_decoding(self):
        # Decoding first would turn %41 into a letter glued to the term, so the term no longer ends at a word edge.
        self.assertEqual(scrubber(["acme corp"])("broken link acme%20corp%41.html"), "broken link [term #1]A.html")

    def test_scrubbing_a_crafted_line_stays_fast(self):
        # A gap such as "< >" matches as one tag or as three characters; unless each gap has only one reading, each
        # one doubles the work, so a crafted broken link could hang the site step until the job times out.
        scrub = scrubber(["acme corp"])
        for gap in ("< >", "<%20>", "<&nbsp;>", "<&#160;>", "<-&nbsp;->", "<_>"):   # seconds to minutes with two readings
            start = time.monotonic()
            scrub("broken link acme" + gap * 24 + "x.html")
            self.assertLess(time.monotonic() - start, 1.0, gap)

    def test_any_tag_shaped_gap_is_hidden(self):
        # A punctuated term has only the loose rule, so every <...> the old rule read as a gap must still be one.
        scrub = scrubber(["acme-corp"])
        for gap in ("<1>", "<!--x-->", "< b>", "</ b>", "<_x>", "<!DOCTYPE html>", "<?xml v?>", "<b>", "< >",
                    "<%1>", "<&1>", "<%20b>", "<%20%3Eb>", "<&NBSP;b>"):
            self.assertNotIn("acme", scrub(f"x acme{gap}corp.png"), gap)

    def test_term_at_the_end_of_a_longer_word_is_left_alone(self):
        self.assertEqual(scrubber(["ann"])("joann and ann"), "joann and [term #1]")

    def test_traceback_is_scrubbed(self):
        with mock.patch("check.check", side_effect=OSError("cannot read posts/acme-corp/index.html")):
            out = self.assert_fails_without_leak({"index.html": PAGE.format(body="")})
        self.assertIn("OSError", out)

    def test_byte_order_mark_does_not_hide_the_first_term(self):
        failed, out = self.run_main({"notes.md": "acme corp"}, "\ufeffacme corp\n")
        self.assertTrue(failed, out)

    def test_show_terms_prints_the_term_for_local_debugging(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("CI", None)
            failed, out = self.run_main({"notes.md": "acme corp"}, "acme corp\n", "--show-terms")
        self.assertTrue(failed)
        self.assertIn("acme corp", out)

    def test_show_terms_is_refused_in_ci(self):
        with mock.patch.dict(os.environ, {"CI": "true"}):
            failed, out = self.run_main({"notes.md": "acme corp"}, "acme corp\n", "--show-terms")
        self.assertTrue(failed)
        self.assertNotRegex(out, LEAK)

    def test_denylist_only_skips_the_site_checks(self):
        failed, out = self.run_main({"index.html": "<a href='missing.html'>x</a>"}, "acme corp\n", "--denylist-only")
        self.assertFalse(failed, out)

    def test_no_terms_fails_closed(self):
        clean = {"index.html": PAGE.format(body=""), "posts/index.html": PAGE.format(body=MUSICGEN)}
        for terms in ("", "\n\n", "# a\n# b\n", "  # note\n\t# other\n"):
            failed, out = self.run_main(clean, terms)
            self.assertTrue(failed, f"{terms!r} loaded as a denylist")
            self.assertIn("no terms", out)


if __name__ == "__main__":
    unittest.main()
