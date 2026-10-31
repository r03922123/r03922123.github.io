# Tests for check.py: a clean site passes; each planted fault is reported.
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

    def test_post_not_listed_on_home(self):
        out = self.findings({"index.html": PAGE.format(body=""), "posts/a/index.html": PAGE.format(body="")})
        self.assertEqual(len(out), 1)
        self.assertIn("posts/a/", out[0])


if __name__ == "__main__":
    unittest.main()
