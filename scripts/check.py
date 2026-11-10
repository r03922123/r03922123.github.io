"""Site checks run before every commit: privacy, internal links, page head, pinned CDNs, post listing.

GitHub Pages serves every file in this repo, so the private-term denylist is scanned across every served
text file and is itself kept out of git (scripts/denylist.local.txt). Usage: python3 scripts/check.py
[--denylist-only] [--show-terms]. Actions logs on this public repo are public, so every output line is scrubbed of
private terms unless --show-terms is given (local debugging only).
"""
import argparse
import os
import re
import sys
import traceback
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
from xml.etree import ElementTree

PAGES = {".html", ".htm"}
SOURCES = "tools/leetcode/src"                    # Markdown that build.py renders into pages
DENYLIST = Path(__file__).with_name("denylist.local.txt")
EXTERNAL = {"musicgen-prune-inference"}           # project repos GitHub Pages serves beside this one, at /<repo>/
MUSICGEN = ("/musicgen-prune-inference/", "/musicgen-prune-inference/decoding/")

# Leak markers that only make sense to look for in rendered pages (and the sources they are rendered from).
HTML_PATTERNS = {
    "phone number": re.compile(r"\+\d{1,3}[ -]?\d{2,4}[ -]?\d{3}[ -]?\d{3,4}"),
    "phone number (national)": re.compile(r"(?<![\d+])09\d{2}[-. ]?\d{3}[-. ]?\d{3}(?!\d)"),
    # a note name such as [[LoRA]]; nested lists like [[1, 2], [3]] or df[["col"]] are not wikilinks
    "vault wikilink": re.compile(r"""\[\[[^\[\]\n"']*[A-Za-z]{2}[^\[\]\n"']*\]\]"""),
    "vault callout": re.compile(r"(?:>|&gt;)\s?\[!"),                # build.py escapes the '>'
    "internal host": re.compile(r"s3://|confluence|\.azure\.|azurewebsites", re.I),
}
HEAD = {
    "lang": re.compile(r"<html[^>]*\slang="),
    "viewport": re.compile(r'name="viewport"'),
    "title": re.compile(r"<title>[^<]+</title>"),
    "description": re.compile(r'name="description"'),
}
LINK = re.compile(r"""(?:href|src)\s*=\s*(?:"([^"]+)"|'([^']+)')""")
CDN = re.compile(r'<script[^>]+src="(https?://[^"]+)"|<link[^>]+href="(https?://[^"]+\.css)"')
PINNED = re.compile(r"[@/]\d+\.\d+\.\d+")
SVG = re.compile(r"<svg\b.*?</svg>", re.S)
SVG_LINK = re.compile(r'(<a\b[^>]*\bhref="([^"]*)"[^>]*>)')
HOME_BUDGET = 150_000
# SPEC-post-feeds: the files tools/feeds/build.py writes, and the pages the sitemap must hold besides the posts.
SITE = "https://r03922123.github.io"
FEEDS = ("rss.xml", "sitemap.xml", "robots.txt", "llms.txt")
FIXED_PAGES = ("/", "/about.html", "/posts/", "/leetcode/")
SITEMAP_LOC = "{http://www.sitemaps.org/schemas/sitemap/0.9}loc"
MD_LINK = re.compile(r"\[[^\]]+\]\(([^)\s]+)\)")    # the link syntax tools/leetcode/build.py renders
MD_CODE = re.compile(r"^```.*?^```|`[^`\n]+`", re.S | re.M)   # fences and code spans, which it renders as no link
EMAIL = re.compile(r"[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")   # not a pinned name@1.2.3
# Hand-written LeetCode post counts on the home page and posts index; build.py only rewrites the series page.
LEETCODE_COUNT = re.compile(r"LeetCode\s+patterns:\s+(\d+)|LEETCODE\s+(?:·|&middot;)\s+(\d+)|\b(\d+)\s+pattern\b"
                            r"|has\s+(\d+)\s+short\s+posts")
PHONE_SEP = r"(?:[-. ()_/\u00a0\u2011]|&#8209;|&#x2011;|&nbsp;)*"   # _ and /: names are scanned with them as gaps
WORD_SEP = r"(?:\s|&nbsp;|&#0*160;|&#x0*a0;|<[^>]+>)+"     # what a browser shows as a space between two words
NAME_SEP = re.compile(r"[-_/.]")                  # how a file name separates words; Pages serves names as URLs
# Scrubbing over-redacts: any gap, or none. Each gap must have exactly one reading, or each one doubles the time and
# a crafted line can hang the run. So a <...> counts as a tag only if it holds a letter or digit that no other branch
# can read: "< >", "<%20>" or "<&nbsp;>" are read piece by piece instead. This hides exactly what <[^>]+> hid.
GAP_CODES = r"%20|&nbsp;|&#0*160;|&#x0*a0;"
NOT_A_WORD = rf"(?:{GAP_CODES}|%(?!20)|&(?!nbsp;|#0*160;|#x0*a0;)|[^\w<>%&]|_)"   # pieces with no readable letter
TAG_GAP = rf"<(?={NOT_A_WORD}*[^\W_])[^>]*>"
LOOSE_SEP = rf"(?:[\W_]|{GAP_CODES}|{TAG_GAP})*"


def term_rx(term):
    """A phone-number term matches in any national, +886 or separated form; other terms as whole words."""
    if sum(ch.isdigit() for ch in term) >= 6 and re.fullmatch(r"[\d\s+().-]+", term):
        core = re.sub(r"\D", "", term).lstrip("0")
        prefix = rf"(?:886{PHONE_SEP})?(?:0{PHONE_SEP})?"  # each separator run follows a digit: no backtracking blowup
        return re.compile(r"(?<!\d)" + prefix + PHONE_SEP.join(core) + r"(?!\d)")
    return re.compile(r"(?<!\w)" + WORD_SEP.join(map(re.escape, term.split())) + r"(?!\w)", re.I)


def term_forms(term):
    """The term as listed, and with its - _ . / read as gaps: names and pages are read that way before matching, so
    acme-corp must also match acme_corp.png and "acme corp"."""
    return [term_rx(form) for form in dict.fromkeys([term, NAME_SEP.sub(" ", term).strip()]) if form]


def scrubber(terms):
    """A function that replaces every form of each term in a line with its denylist number, longest term first so
    that a term inside a longer one cannot leave the rest of the longer one showing."""
    rules = []
    for n, term in sorted(enumerate(terms, 1), key=lambda item: -len(item[1])):
        words = re.findall(r"[^\W_]+", term)
        # Word edges: a label inside "planning" would mark "ann" as private beside public text that fills the gap.
        loose = [re.compile(r"(?<![^\W_])" + LOOSE_SEP.join(map(re.escape, words)) + r"(?![^\W_])", re.I)] if words else []
        rules += [(rx, f"[term #{n}]") for rx in [term_rx(term), *loose]]

    def redact(line):
        for rx, label in rules:
            line = rx.sub(label, line)
        return line

    def scrub(line):                                  # raw forms first, then the decoded line, so neither hides
        return redact(unquote(redact(line)))          # the other; a %-encoded term is printed decoded and redacted
    return scrub


def text_files(root):
    """Every file Pages would serve that holds text; binary files (any NUL byte), such as images, are skipped."""
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".git" not in path.parts and path.name != DENYLIST.name:
            data = path.read_bytes()
            if b"\0" not in data:
                yield path, data.decode(errors="replace")


def exists_exactly(root, rel):
    # Pages is case-sensitive while macOS disks are not, so compare each part against the real listing.
    path = root
    for part in rel.parts:
        if not path.is_dir() or part not in os.listdir(path):
            return False
        path = path / part
    return path.is_file()


def link_ok(root, page, url):
    if re.match(r"[a-z]+:|//|#", url):
        return True
    url = url.split("#")[0].split("?")[0]
    if url.startswith("/"):
        if url.strip("/").split("/")[0] in EXTERNAL:
            return True
        target = url.lstrip("/")
    else:
        target = (page.parent.relative_to(root) / url).as_posix()
    target = Path(os.path.normpath(target))
    if target.parts[:1] == ("..",):
        return False                                  # above the site root
    return exists_exactly(root, target) or exists_exactly(root, target / "index.html")


class Anchors(HTMLParser):
    """Ids and real <a href> elements in document order (comments and script text are skipped), noting the
    links inside the home's .lofi-latest list."""

    def __init__(self, text):
        super().__init__()
        self.items, self.latest, self.depth = [], None, 0
        self.feed(text)
        self.close()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == self.latest:
            self.depth += 1
        elif self.latest is None and "lofi-latest" in (attrs.get("class") or "").split():
            self.latest, self.depth = tag, 1
        if attrs.get("id"):
            self.items.append(("id", attrs["id"], False))
        if tag == "a" and attrs.get("href") is not None:
            self.items.append(("a", attrs["href"], self.latest is not None))

    def handle_endtag(self, tag):
        if tag == self.latest:
            self.depth -= 1
            if not self.depth:
                self.latest = None


def site_path(base, href):
    """Where an href on a page at `base` points: a path on this site, or the full URL for other schemes and hosts."""
    joined = urljoin(base, href)
    url = urlsplit(joined)
    return joined if url.scheme or url.netloc else re.sub(r"index\.html?$", "", url.path)


def links(root, rel, base):
    """Anchors(...).items of the page at rel (served at base), each href resolved with site_path."""
    page = root / rel
    items = Anchors(page.read_text(errors="replace")).items if page.exists() else []
    return [(kind, site_path(base, value) if kind == "a" else value, latest) for kind, value, latest in items]


def feed_findings(root, listed):
    """SPEC-post-feeds. Active once tools/feeds/build.py exists, so a deleted or broken feed fails instead of passing."""
    if not (root / "tools/feeds/build.py").is_file():
        return []
    findings = [f"{name}: missing" for name in FEEDS if not (root / name).is_file()]
    trees = {}
    for name in ("rss.xml", "sitemap.xml"):
        if (root / name).is_file():
            try:
                trees[name] = ElementTree.parse(root / name).getroot()
            except ElementTree.ParseError as error:        # the message gives a line and column, never page text
                findings.append(f"{name}: malformed XML ({error})")
    if "sitemap.xml" in trees:
        locs = {(loc.text or "").strip() for loc in trees["sitemap.xml"].iter(SITEMAP_LOC)}
        findings += [f"sitemap.xml: missing {SITE}{path}" for path in listed if SITE + path not in locs]
    if "rss.xml" in trees:
        findings += [f"rss.xml: item {n} has no pubDate" for n, item in enumerate(trees["rss.xml"].iter("item"), 1)
                     if not (item.findtext("pubDate") or "").strip()]
    for name in FEEDS:                                # RSS author is an email; this site's feeds carry none
        if (root / name).is_file():
            lines = (root / name).read_text(errors="replace").splitlines()
            findings += [f"{name}:{n}: email address" for n, line in enumerate(lines, 1) if EMAIL.search(line)]
    for section in ("posts", "leetcode"):             # each source gets a Markdown copy with # title, no front matter
        for src in sorted((root / "tools" / section / "src").glob("*.md")):
            md = root / section / src.stem / "index.md"
            if not md.is_file():
                findings.append(f"{section}/{src.stem}/index.md: missing (source {src.relative_to(root).as_posix()})")
                continue
            text = md.read_text(errors="replace")
            if text.startswith("---"):
                findings.append(f"{section}/{src.stem}/index.md: front matter kept")
            # The page drops a link to an unwritten post (build.py's link()); the Markdown copy keeps it live.
            findings += [f"{section}/{src.stem}/index.md: broken link {url}" for url in MD_LINK.findall(MD_CODE.sub("", text))
                         if not link_ok(root, md, url)]
    return findings


def check(root, denylist, terms_only=False):
    root = Path(root)
    deny = [(term, term_forms(term)) for term in denylist]
    findings = []
    # Names too, binary files included: the builds print paths and Pages serves names, so acme-corp.png leaks.
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".git" not in path.parts:
            rel = path.relative_to(root).as_posix()
            findings += [f"{rel}: private term in file name {term!r}" for term, rxs in deny
                         if any(rx.search(name) for rx in rxs for name in (unquote(rel), NAME_SEP.sub(" ", unquote(rel))))]
    for path, text in text_files(root):
        rel = path.relative_to(root).as_posix()
        decoded = unquote(text)                       # a link such as /%61cme%20corp/ is the term once served
        # Each also with file-name gaps (acme-corp quoted in a page); NAME_SEP swaps one char for one, keeping lines.
        bodies = {b for body in (text, decoded) for b in (body, NAME_SEP.sub(" ", body))}
        for term, rxs in deny:                        # whole text, so a term split across lines is still found
            lines = sorted({body.count("\n", 0, m.start()) + 1 for body in bodies for rx in rxs for m in rx.finditer(body)})
            findings += [f"{rel}:{n}: private term {term!r}" for n in lines]
        if terms_only:
            continue
        if path.suffix in PAGES or (path.suffix == ".md" and path.parent.relative_to(root).as_posix() == SOURCES):
            for n, line in enumerate(text.splitlines(), 1):
                findings += [f"{rel}:{n}: {name}" for name, rx in HTML_PATTERNS.items() if rx.search(line)]
        if path.suffix not in PAGES:
            continue
        findings += [f"{rel}: missing {name}" for name, rx in HEAD.items() if not rx.search(text)]
        findings += [f"{rel}: broken link {url}" for url in (a or b for a, b in LINK.findall(text))
                     if not link_ok(root, path, url)]
        for svg in SVG.findall(text):                 # scene doors: links drawn in SVG must still be named
            for tag, href in SVG_LINK.findall(svg):
                if "aria-label=" not in tag:
                    findings.append(f"{rel}: link inside SVG without aria-label: {href}")
        for script, style in CDN.findall(text):
            if not PINNED.search(script or style):
                findings.append(f"{rel}: unpinned CDN {script or style}")
    if terms_only:
        return findings
    home = root / "index.html"
    if home.exists():                                 # SPEC-scene: the home page (HTML + stylesheet) stays light
        weight = home.stat().st_size + ((root / "style.css").stat().st_size if (root / "style.css").exists() else 0)
        if weight > HOME_BUDGET:
            findings.append(f"index.html: home weighs {weight / 1000:.0f} KB with style.css (budget {HOME_BUDGET // 1000} KB)")
    # Each post folder must be linked from its listing page: posts/ from the posts index, leetcode/ from its series page.
    for section in ("posts", "leetcode"):
        linked = {path for kind, path, _ in links(root, f"{section}/index.html", f"/{section}/") if kind == "a"}
        for post in sorted(root.glob(f"{section}/*/index.html")):
            if f"/{section}/{post.parent.name}/" not in linked:
                findings.append(f"{section}/index.html: {section}/{post.parent.name}/ is not listed")
    # SPEC-reskin: the posts index also lists the MusicGen posts, under #musicgen (the radio's door).
    listing = links(root, "posts/index.html", "/posts/")
    if (root / "posts/index.html").exists():
        start = next((i for i, item in enumerate(listing) if item[:2] == ("id", "musicgen")), None)
        if start is None:
            findings.append("posts/index.html: missing the #musicgen section")
        else:
            after = {path for kind, path, _ in listing[start:] if kind == "a"}
            findings += [f"posts/index.html: #musicgen does not link {url}" for url in MUSICGEN if url not in after]
    # Every post a listing links, MusicGen included, must be in the sitemap.
    listed = [path for section in ("posts", "leetcode") for kind, path, _ in links(root, f"{section}/index.html", f"/{section}/")
              if kind == "a" and (re.fullmatch(rf"/{section}/[^/]+/", path) or path in MUSICGEN)]
    findings += feed_findings(root, list(dict.fromkeys([*FIXED_PAGES, *listed])))
    # The home page's hand-written copies of the listings must keep up with them.
    count = len(list(root.glob("leetcode/*/index.html")))
    for rel in ("index.html", "posts/index.html"):
        text = (root / rel).read_text(errors="replace") if (root / rel).exists() else ""
        for m in LEETCODE_COUNT.finditer(text):
            said, n = int(next(g for g in m.groups() if g)), text.count("\n", 0, m.start()) + 1
            if said != count:
                findings.append(f"{rel}:{n}: says {said} LeetCode posts, leetcode/ has {count}")
    latest = [path for kind, path, in_latest in links(root, "index.html", "/") if kind == "a" and in_latest]
    if latest:                                        # SPEC-lofi: the plain home shows the latest three posts
        posts = [path for kind, path, _ in listing if kind == "a" and re.fullmatch(r"/posts/[^/]+/", path)]
        top = list(dict.fromkeys(posts))[:3]
        if latest != top:
            findings.append(f"index.html: .lofi-latest lists {', '.join(latest)}; the posts index starts {', '.join(top)}")
    return findings


def main(argv=None, root=None, denylist=DENYLIST):
    parser = argparse.ArgumentParser(description="Site checks; output is scrubbed of private terms.")
    parser.add_argument("--denylist-only", action="store_true", help="only the private-term scan of contents and names")
    parser.add_argument("--show-terms", action="store_true", help="print terms unscrubbed (local debugging only)")
    args = parser.parse_args(argv)
    if args.show_terms and os.environ.get("CI"):
        sys.exit("--show-terms is refused in CI: Actions logs on this repo are public")
    if not denylist.exists():
        sys.exit(f"missing {denylist}: create it with one private term per line (see README)")
    # Strip before testing for '#', so an indented comment is not loaded as a term.
    terms = [t for t in (line.strip() for line in denylist.read_text(encoding="utf-8-sig").splitlines()) if t and not t.startswith("#")]
    if not terms:                                     # an unset CI secret writes an empty file: fail closed
        sys.exit(f"no terms in {denylist.name}: the privacy check would pass vacuously")
    scrub = (lambda line: line) if args.show_terms else scrubber(terms)
    try:
        findings = check(root or Path(__file__).resolve().parent.parent, terms, terms_only=args.denylist_only)
    except Exception:                                 # a traceback can quote a path or page text: scrub it too
        findings = traceback.format_exc().splitlines()
    print("\n".join(map(scrub, findings)) or "ok")
    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
