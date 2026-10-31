"""Site checks run before every commit: privacy, internal links, page head, pinned CDNs, post listing.

GitHub Pages serves every file in this repo, so the private-term denylist is scanned across all text
files and is itself kept out of git (scripts/denylist.local.txt). Usage: python3 scripts/check.py
"""
import re
import sys
from pathlib import Path

TEXT = {".html", ".md", ".css", ".js", ".py", ".txt", ".json", ".xml", ".svg"}
DENYLIST = Path(__file__).with_name("denylist.local.txt")

# Leak markers that only make sense to look for in rendered pages.
HTML_PATTERNS = {
    "phone number": re.compile(r"\+\d{1,3}[ -]?\d{2,4}[ -]?\d{3}[ -]?\d{3,4}"),
    # a note name such as [[LoRA]]; nested lists like [[1, 2], [3]] or df[["col"]] are not wikilinks
    "vault wikilink": re.compile(r"""\[\[[^\[\]\n"']*[A-Za-z]{2}[^\[\]\n"']*\]\]"""),
    "vault callout": re.compile(r">\s?\[!"),
    "internal host": re.compile(r"s3://|confluence|\.azure\.|azurewebsites", re.I),
}
HEAD = {
    "lang": re.compile(r"<html[^>]*\slang="),
    "viewport": re.compile(r'name="viewport"'),
    "title": re.compile(r"<title>[^<]+</title>"),
    "description": re.compile(r'name="description"'),
}
LINK = re.compile(r'(?:href|src)="([^"]+)"')
CDN = re.compile(r'<script[^>]+src="(https?://[^"]+)"|<link[^>]+href="(https?://[^"]+\.css)"')
PINNED = re.compile(r"[@/]\d+\.\d+\.\d+")


def text_files(root):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in TEXT and ".git" not in path.parts and path.name != DENYLIST.name:
            yield path


def link_ok(root, page, url):
    if re.match(r"[a-z]+:|//|#", url):
        return True
    url = url.split("#")[0].split("?")[0]
    if url.startswith("/"):
        first = url.strip("/").split("/")[0]
        if first and "." not in first and not (root / first).exists():
            return True  # another project repo served beside this one, e.g. /musicgen-prune-inference/
        target = root / url.lstrip("/")
    else:
        target = page.parent / url
    return target.is_file() or (target / "index.html").is_file()


def check(root, denylist):
    root = Path(root)
    deny = [(term, re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.I)) for term in denylist]
    findings = []
    for path in text_files(root):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(errors="replace")
        for n, line in enumerate(text.splitlines(), 1):
            findings += [f"{rel}:{n}: private term {term!r}" for term, rx in deny if rx.search(line)]
            if path.suffix == ".html":
                findings += [f"{rel}:{n}: {name}" for name, rx in HTML_PATTERNS.items() if rx.search(line)]
        if path.suffix != ".html":
            continue
        findings += [f"{rel}: missing {name}" for name, rx in HEAD.items() if not rx.search(text)]
        findings += [f"{rel}: broken link {url}" for url in LINK.findall(text) if not link_ok(root, path, url)]
        for script, style in CDN.findall(text):
            if not PINNED.search(script or style):
                findings.append(f"{rel}: unpinned CDN {script or style}")
    # Each post folder must be linked from its listing page: posts/ from the home page, leetcode/ from its series page.
    for section, listing in (("posts", "index.html"), ("leetcode", "leetcode/index.html")):
        page = (root / listing).read_text() if (root / listing).exists() else ""
        for post in sorted(root.glob(f"{section}/*/index.html")):
            href = f"{section}/{post.parent.name}/"
            if href not in page and f'"{post.parent.name}/"' not in page:
                findings.append(f"{listing}: {href} is not listed")
    return findings


def main():
    if not DENYLIST.exists():
        sys.exit(f"missing {DENYLIST}: create it with one private term per line (see README)")
    terms = [t.strip() for t in DENYLIST.read_text().splitlines() if t.strip() and not t.startswith("#")]
    findings = check(Path(__file__).resolve().parent.parent, terms)
    print("\n".join(findings) or "ok")
    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
