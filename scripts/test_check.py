# Tests for check.py: a clean site passes; each planted fault is reported.
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


class CheckTest(unittest.TestCase):
    def site(self, files):
        root = Path(tempfile.mkdtemp())
        for path, text in files.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(text)
        return root

    def findings(self, files):
        return check(self.site(files), DENY)

    def test_clean_site_passes(self):
        files = {
            "index.html": PAGE.format(body='<a href="posts/a/">a</a> <a href="https://x.org">x</a>'),
            "posts/index.html": PAGE.format(body='<a href="a/">a</a>'),
            "posts/a/index.html": PAGE.format(body='<a href="../../">home</a>'),
            "style.css": "body{}",
        }
        self.assertEqual(self.findings(files), [])

    def test_denylist_term_in_any_text_file(self):
        out = self.findings({"index.html": PAGE.format(body=""), "notes.md": "has Secret-Term here"})
        self.assertEqual(len(out), 1)
        self.assertIn("notes.md:1", out[0])

    def test_denylist_matches_whole_words_only(self):
        out = check(self.site({"index.html": PAGE.format(body="Cacmeal\nthen Acme.")}), ["acme"])
        self.assertEqual(out, [f"index.html:{PAGE.count(chr(10)) + 1}: private term 'acme'"])

    def test_phone_numbers(self):
        out = self.findings({"index.html": PAGE.format(body="call 555-0100 or +886 905 123 456")})
        self.assertEqual(len(out), 2)

    def test_vault_syntax_in_html(self):
        out = self.findings({"index.html": PAGE.format(body="see [[note]]\n> [!tip] x")})
        self.assertEqual(len(out), 2)

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
                 "posts/index.html": PAGE.format(body=""),
                 "posts/a/index.html": PAGE.format(body="")}
        self.assertEqual(self.findings(files), ["posts/index.html: posts/a/ is not listed"])

    def test_post_not_listed_on_home(self):
        out = self.findings({"index.html": PAGE.format(body=""), "posts/a/index.html": PAGE.format(body="")})
        self.assertEqual(len(out), 1)
        self.assertIn("posts/a/", out[0])


if __name__ == "__main__":
    unittest.main()


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
            for text in ("fg", "muted", "link", "accent"):
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
