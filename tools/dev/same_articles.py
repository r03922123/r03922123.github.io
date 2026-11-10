"""same_articles.py <git-ref>: article text of every reading page must match <git-ref> (chrome may differ)."""
import html, re, subprocess, sys
from pathlib import Path

def text(page):
    m = re.search(r"<article\b.*?</article>", page, re.S) or re.search(r"<main\b.*?</main>", page, re.S)
    body = m.group(0) if m else ""                 # chrome the shell moves: title block, side column, contents, crumb
    body = re.sub(r"<(header|aside|nav)\b.*?</\1>|<p class=\"crumb\">.*?</p>", " ", body, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", body)
    return " ".join(html.unescape(t).split())

ref, changed, n = sys.argv[1], [], 0
for path in sorted([*Path("posts").glob("*/index.html"), *Path("leetcode").glob("*/index.html"), Path("about.html")]):
    try:
        before = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        continue                                   # new page, nothing to compare
    n += 1
    if text(before) != text(path.read_text()):
        changed.append(str(path))
print(f"{n} pages compared; changed: {changed or 'none'}")
sys.exit(1 if changed else 0)
