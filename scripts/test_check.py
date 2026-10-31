# Tests for check.py: a clean site passes; each planted fault is reported.
import ast
import re
import tempfile
import unittest
from pathlib import Path

from check import check

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
        body = '<svg viewBox="0 0 10 10"><a href="#x" aria-label="X"><rect/></a><a href="#y"><rect/></a></svg>'
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

    def test_main_block_comes_last(self):
        # unittest.main() exits, so any test class defined after it would not run when this file is executed
        last = ast.parse(Path(__file__).read_text()).body[-1]
        self.assertIn("__main__", ast.unparse(last.test) if isinstance(last, ast.If) else "")


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


class ThemeTest(unittest.TestCase):
    """SPEC-theme: text tokens meet WCAG AA on every background, in both variants; colours live in tokens."""

    def variants(self):
        css = STYLE.read_text()
        light = css[css.index(":root {"):css.index("}", css.index(":root {"))]
        dark_start = css.index("@media (prefers-color-scheme: dark)")
        dark = css[dark_start:css.index("}", css.index(":root {", dark_start))]
        return {"paper": tokens(light), "night": tokens(dark)}

    def test_text_tokens_meet_aa(self):
        for name, t in self.variants().items():
            for text in ("fg", "muted", "link", "accent", "code-builtin", "code-keyword"):
                for surface in ("bg", "card"):
                    ratio = contrast(t[text], t[surface])
                    self.assertGreaterEqual(ratio, 4.5, f"{name}: --{text} on --{surface} is {ratio:.2f}:1")

    def test_no_hard_coded_colours_outside_tokens_and_scene(self):
        css = STYLE.read_text()
        dark_start = css.index("@media (prefers-color-scheme: dark)")
        body = css[css.index("}", css.index(":root {", dark_start)) + 1:]        # after both token blocks
        body = body.split("/* Scene", 1)[0]                                       # scene art may use literal colours
        stray = re.findall(r"#[0-9a-fA-F]{3,8}\b|rgba?\(", body)
        self.assertEqual(stray, [], "colours outside the token blocks")


if __name__ == "__main__":
    unittest.main()
