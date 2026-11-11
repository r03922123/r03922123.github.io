# Tests for voice_check.py: one failing case per rule in SPEC-post-voice.md, plus the cases its own choices add.
import shutil
import tempfile
import unittest
from pathlib import Path

from voice_check import check, snapshot, write_baseline

MD = """---
title: Dropout
summary: What dropout does.
group: Machine learning fundamentals
order: 8
tags: regularization
---
Dropout sets each unit's output to zero with probability \\(p\\) and uses `nn.Dropout` in PyTorch.
Training error falls from 0.03 to 0.13 over two runs, see [the paper](https://arxiv.org/abs/1207.0580).

## Results
| p | test error |
|---|---|
| 0.5 | 0.12 |

```python
seeds = 4
assert seeds == 4
```

\\[ \\tilde h = \\frac{m \\odot h}{1 - p} \\]

1. A list item that says nothing numeric at all here.
"""
PAGE = """<html><body><header class="site"><nav><a href="/">Home</a></nav></header>
<article class="post-body">
<header class="post-head"><h1>Position bias</h1><div class="meta">October 31, 2026 · 6 min read</div></header>
<p>Ten items with known relevance, and raw CTR at rank 1 is <code>0.5</code> of \\(p_k\\).</p>
<nav class="toc"><h2>Contents</h2><ol><li><a href="#ips">IPS</a></li></ol></nav>
<h2 id="ips">Inverse propensity weighting</h2>
<p>Examination falls off as \\(p_k = 1/k\\) over 3 seeds <a href="#ref-1">[1]</a>.</p>
<pre><code class="language-python">n = 10000</code></pre>
<table><tr><td>IPS</td><td>0.93</td></tr></table>
<ol class="refs"><li id="ref-1">T. Joachims. SIGIR 2005.</li></ol>
</article></body></html>
"""


class VoiceCheckTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        (self.root / "scripts").mkdir()
        self.src = self.root / "tools/posts/src/dropout.md"
        self.page = self.root / "posts/position-bias/index.html"
        for path, text in ((self.src, MD), (self.page, PAGE)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        (self.root / "posts/dropout").mkdir()            # a generated page beside its source is not long-form
        (self.root / "posts/dropout/index.html").write_text(PAGE.replace("Ten", "Nine"))
        write_baseline(self.root)

    def edit(self, path, old, new):
        text = path.read_text()
        self.assertIn(old, text)
        path.write_text(text.replace(old, new, 1))

    def errors(self, final=False):
        return check(self.root, final)

    def assertFails(self, rule, final=False):
        errors = self.errors(final)
        self.assertTrue(any(rule in e for e in errors), f"no '{rule}' in {errors}")

    def test_unchanged_post_passes_and_fails_final(self):
        self.assertEqual(self.errors(), [])
        self.assertEqual(sorted(snapshot(self.root)), ["posts/position-bias/index.html", "tools/posts/src/dropout.md"])
        self.assertFails("prose unchanged", final=True)

    def test_reworded_post_passes_final(self):
        self.edit(self.src, "Dropout sets", "Dropout, in short, sets")
        self.edit(self.page, "Ten items", "Ten items, all")
        self.assertEqual(self.errors(final=True), [])

    def test_changed_code_block_fails(self):
        self.edit(self.src, "assert seeds == 4", "assert seeds == 4  # four seeds")
        self.assertFails("code blocks changed")
        self.edit(self.page, "n = 10000", "n = 10_000")
        self.assertFails("posts/position-bias/index.html: code blocks changed")

    def test_changed_table_fails(self):
        self.edit(self.src, "| 0.5 | 0.12 |", "| 0.5 | 0.12  |")
        self.assertFails("tables changed")
        self.edit(self.page, "<td>0.93</td>", "<td>0.930</td>")
        self.assertFails("posts/position-bias/index.html: tables changed")

    def test_new_number_in_prose_fails(self):
        self.edit(self.src, "over two runs", "over two runs on 17 tasks")
        self.assertFails("new numbers in prose: 17")

    def test_new_number_word_fails(self):
        self.edit(self.src, "over two runs", "over two runs on five tasks")
        self.assertFails("new numbers in prose: 5")
        self.edit(self.src, "five tasks", "fifteen tasks")
        self.assertFails("new numbers in prose: 15")

    def test_leading_dot_decimal_is_a_number(self):
        self.edit(self.src, "over two runs", "over two runs at .7")
        self.assertFails("new numbers in prose: 0.7")

    def test_entities_in_evidence_are_not_numbers(self):  # &#39; in a code block must not license a 39 in prose
        self.edit(self.page, "n = 10000", "n = &#39;10000&#39;")
        write = self.root / "scripts/voice-baseline.json"
        write.unlink()
        write_baseline(self.root)
        self.edit(self.page, "over 3 seeds", "over 3 seeds and 39 days")
        self.assertFails("new numbers in prose: 39")

    def test_dropped_number_fails(self):
        self.edit(self.src, "falls from 0.03 to 0.13", "falls")
        self.assertFails("numbers dropped from prose: 0.03, 0.13")
        self.edit(self.page, "over 3 seeds ", "")
        self.assertFails("posts/position-bias/index.html: numbers dropped from prose: 3")

    def test_number_from_code_or_table_may_appear_in_prose(self):
        numeric = lambda: [e for e in self.errors() if "numbers" in e]   # the tiny fixture also trips "prose grew"
        self.edit(self.src, "over two runs", "over two runs and 4 seeds, test error 0.12 at p 0.5")
        self.assertEqual(numeric(), [])
        self.edit(self.page, "over 3 seeds", "over 3 seeds and 10,000 sessions, IPS 0.93")
        self.assertEqual(numeric(), [])

    def test_prose_growth_over_ten_percent_fails(self):
        words = snapshot(self.root)["tools/posts/src/dropout.md"]["words"]
        self.edit(self.src, "in PyTorch.", "in PyTorch." + " word" * (words // 10))
        self.assertEqual(self.errors(), [])                  # the largest count 10% allows
        self.edit(self.src, "in PyTorch.", "in PyTorch. word")
        self.assertFails("prose grew")

    def test_frozen_inline_code_math_headings_and_links(self):
        for old, new, rule in (("`nn.Dropout`", "`torch.nn.Dropout`", "inline code changed"),
                               ("\\(p\\)", "\\(q\\)", "maths changed"),
                               ("\\frac{m", "\\frac{ m", "maths changed"),
                               ("## Results", "## What we found", "headings, front matter, references or links changed"),
                               ("summary: What", "summary: How", "headings, front matter, references or links changed"),
                               ("1207.0580", "1207.0581", "headings, front matter, references or links changed")):
            self.edit(self.src, old, new)
            self.assertFails(rule)
            self.edit(self.src, new, old)
        self.assertEqual(self.errors(), [])

    def test_long_form_prose_skips_header_nav_and_aside(self):
        self.edit(self.page, "6 min read", "7 min read")
        self.edit(self.page, '<li><a href="#ips">IPS</a></li>', '<li><a href="#ips">IPS 2</a></li>')
        self.edit(self.page, "</article>", '<aside class="post-side">Note 99</aside></article>')
        self.assertEqual(self.errors(), [])

    def test_other_aside_is_prose(self):                  # only reading-shell's own header, nav and aside are skipped
        self.edit(self.page, "</article>", "<aside>Note 99</aside></article>")
        self.assertFails("posts/position-bias/index.html: new numbers in prose: 99")

    def test_long_form_frozen_spans_headings_links_and_refs(self):
        for old, new, rule in (("<code>0.5</code>", "<code>0.50</code>", "inline code changed"),
                               ("\\(p_k = 1/k\\)", "\\(p_k = 1/k^2\\)", "maths changed"),
                               ('<h2 id="ips">Inverse', '<h2 id="ips">The inverse', "headings, front matter, references or links changed"),
                               ('href="#ref-1"', 'href="#ref-2"', "headings, front matter, references or links changed"),
                               ("SIGIR 2005.", "SIGIR 2006.", "headings, front matter, references or links changed")):
            self.edit(self.page, old, new)
            self.assertFails("posts/position-bias/index.html: " + rule)
            self.edit(self.page, new, old)
        self.assertEqual(self.errors(), [])

    def test_number_from_a_heading_or_reference_may_appear_in_prose(self):
        self.edit(self.page, "Ten items", "2005: ten items")    # 2005 is only in the reference list
        self.assertEqual(self.errors(), [])

    def test_repeating_a_span_or_link_passes(self):       # a sentence on a table may reuse a symbol or a citation
        self.edit(self.src, "Training error", "At \\(p\\) training error")
        self.edit(self.src, "over two runs", "over [two runs](https://arxiv.org/abs/1207.0580)")
        self.edit(self.page, "Ten items", "Ten items at \\(p_k\\)")
        self.assertEqual(self.errors(), [])

    def test_moving_inline_spans_with_their_sentence_passes(self):
        self.edit(self.src, "Dropout sets each unit's output to zero with probability \\(p\\) and uses `nn.Dropout` "
                  "in PyTorch.\n", "")
        self.edit(self.src, "\n\n## Results", " Dropout sets each unit's output to zero with probability \\(p\\) and "
                  "uses `nn.Dropout` in PyTorch.\n\n## Results")
        self.assertEqual(self.errors(), [])

    def test_missing_or_extra_post_fails(self):
        self.src.rename(self.src.with_name("renamed.md"))
        self.assertFails("tools/posts/src/dropout.md: missing")
        self.assertFails("tools/posts/src/renamed.md: not in the baseline")

    def test_baseline_is_written_once(self):
        with self.assertRaises(SystemExit):
            write_baseline(self.root)


if __name__ == "__main__":
    unittest.main()
