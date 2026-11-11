"""Site checks run before every commit: privacy, internal links, page head, pinned CDNs, post listing.

GitHub Pages serves every file in this repo, so the private-term denylist is scanned across every served
text file and is itself kept out of git (scripts/denylist.local.txt). Usage: python3 scripts/check.py
[--denylist-only] [--show-terms]. Actions logs on this public repo are public, so every output line is scrubbed of
private terms unless --show-terms is given (local debugging only).
"""
import argparse
import json
import os
import re
import sys
import traceback
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
from xml.etree import ElementTree

PAGES = {".html", ".htm"}
SOURCES = ("tools/leetcode/src", "tools/book/src")   # Markdown the builds render into pages
DENYLIST = Path(__file__).with_name("denylist.local.txt")
EXTERNAL = {"musicgen-prune-inference"}           # project repos GitHub Pages serves beside this one, at /<repo>/
MUSICGEN = ("/musicgen-prune-inference/", "/musicgen-prune-inference/decoding/")
# SPEC-resume-book: unlisted on purpose, reached only from the room's old book (ADR 0003).
BOOK = "/book/"
NOINDEX = re.compile(r'<meta\s+name="robots"\s+content="[^"]*\bnoindex\b')

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
# Fences, code spans and inline maths render as no link; maths stops at a blank line, as inline() sees one paragraph.
MD_CODE = re.compile(r"^```.*?^```|`[^`\n]+`|\\\((?:(?!\n\n).)+?\\\)", re.S | re.M)
FONTS = ("/fonts/source-serif-4/source-serif-4-latin-wght-normal.woff2", "/fonts/inter/inter-latin-wght-normal.woff2")
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


def local(url):
    """True for a link this site serves: not another scheme or host, and not a project repo beside this one."""
    path = url.split("#")[0].split("?")[0]
    return not re.match(r"[a-z]+:|//", url) and not (path.startswith("/") and path.strip("/").split("/")[0] in EXTERNAL)


def target(root, page, url):
    """The file a local url on page points to, relative to root, or None if there is none."""
    url = url.split("#")[0].split("?")[0]
    if not url:
        return page.relative_to(root)                 # same page
    rel = url.lstrip("/") if url.startswith("/") else (page.parent.relative_to(root) / url).as_posix()
    rel = Path(os.path.normpath(rel))
    if rel.parts[:1] == ("..",):
        return None                                   # above the site root
    return next((t for t in (rel, rel / "index.html") if exists_exactly(root, t)), None)


def link_ok(root, page, url):
    return url.startswith("#") or not local(url) or target(root, page, url) is not None


def fragment_ok(root, page, url, ids):
    """A local #fragment must name an id on its page; "" and "top" scroll to the top, as browsers do with no id.
    ids caches each page's ids. A missing page is link_ok's finding, and a non-HTML file has no ids to check."""
    fragment = unquote(url.partition("#")[2])
    if "#" not in url or not local(url) or fragment.lower() in ("", "top"):
        return True
    rel = target(root, page, url)
    if rel is None or rel.suffix not in PAGES:
        return True
    if rel not in ids:
        ids[rel] = {value for kind, value, _ in Anchors((root / rel).read_text(errors="replace")).items if kind == "id"}
    return fragment in ids[rel]


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


class Topics(HTMLParser):
    """SPEC-posts-index: the count text (the <span>) of each card in nav.topics, by href, and the number of rows (<li>)
    in each section.group, by id. shortcut: assumes navs and sections are not nested, as the two templates write them."""

    def __init__(self, text):
        super().__init__()
        self.cards, self.rows, self.nav, self.card, self.section, self.count = {}, {}, False, None, None, False
        self.feed(text)
        self.close()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = (attrs.get("class") or "").split()
        if tag == "nav" and "topics" in classes:
            self.nav = True
        elif tag == "a" and self.nav:
            self.card = attrs.get("href") or ""
            self.cards[self.card] = ""
        elif tag == "section" and "group" in classes:
            self.section = attrs.get("id") or ""
            self.rows[self.section] = 0
        elif tag == "li" and self.section is not None:
            self.rows[self.section] += 1
        elif tag == "span" and self.card is not None:
            self.count = True

    def handle_endtag(self, tag):
        if tag == "nav":
            self.nav = False
        elif tag == "a":
            self.card = None
        elif tag == "section":
            self.section = None
        elif tag == "span":
            self.count = False

    def handle_data(self, data):
        if self.card is not None and self.count:
            self.cards[self.card] += data


def topic_findings(rel, text):
    """Every section.group on a list page has a card, and the count the card says is its number of rows."""
    topics, findings = Topics(text), []
    for sid, rows in topics.rows.items():
        card = topics.cards.get(f"#{sid}")
        said = re.search(r"(\d+)\s+posts?\s*$", card or "")
        if card is None:
            findings.append(f"{rel}: section #{sid} has no topic card")
        elif not said or int(said.group(1)) != rows:
            findings.append(f"{rel}: card #{sid} says {said.group(0).strip() if said else 'no count'}, its section has {rows} rows")
    return findings


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


class HeadTags(HTMLParser):
    """A page's <link> attributes and the text of each JSON-LD script."""

    def __init__(self, text):
        super().__init__()
        self.links, self.ld, self.in_ld = [], [], False
        self.feed(text)
        self.close()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "link":
            self.links.append(attrs)
        self.in_ld = tag == "script" and attrs.get("type") == "application/ld+json"
        if self.in_ld:
            self.ld.append("")

    def handle_endtag(self, tag):
        self.in_ld = False

    def handle_data(self, data):
        if self.in_ld:
            self.ld[-1] += data

    def hrefs(self, rel, kind=None):
        return [a.get("href") for a in self.links if rel in (a.get("rel") or "").split() and kind in (None, a.get("type"))]


def head_findings(root):
    """SPEC-post-feeds page head: each page names its own absolute URL and the feed; each post carries JSON-LD, and
    each post with a Markdown source links that copy."""
    pages = [(page, path, None) for page, path in zip(("index.html", "about.html", "posts/index.html", "leetcode/index.html"), FIXED_PAGES)]
    pages += [(p.relative_to(root).as_posix(), f"/{section}/{p.parent.name}/", section)
              for section in ("posts", "leetcode") for p in sorted(root.glob(f"{section}/*/index.html"))]
    findings = []
    for rel, path, section in pages:
        if not (root / rel).is_file():
            continue
        tags = HeadTags((root / rel).read_text(errors="replace"))
        canonical = tags.hrefs("canonical")
        if not canonical:
            findings.append(f"{rel}: no canonical link")
        elif canonical != [SITE + path]:                 # exactly one, and this page's own URL
            findings.append(f"{rel}: canonical link {', '.join(map(str, canonical))} is not {SITE + path}")
        if SITE + "/rss.xml" not in tags.hrefs("alternate", "application/rss+xml"):
            findings.append(f"{rel}: no feed link")
        for text in tags.ld:
            try:
                json.loads(text)
            except json.JSONDecodeError as error:      # line and column only, never page text
                findings.append(f"{rel}: JSON-LD does not parse (line {error.lineno}, column {error.colno})")
            if EMAIL.search(text):
                findings.append(f"{rel}: email address in JSON-LD")
        if section is None:
            continue
        if not tags.ld:
            findings.append(f"{rel}: no JSON-LD")
        sourced = (root / "tools" / section / "src" / f"{Path(rel).parent.name}.md").is_file()
        if sourced and SITE + path + "index.md" not in tags.hrefs("alternate", "text/markdown"):
            findings.append(f"{rel}: no Markdown alternate")
    return findings


def shell_findings(rel, text):
    """SPEC-reading-shell: each page on the stylesheet but the home preloads both reading fonts, and each article
    has its side and text columns, so a hand-written page cannot keep the old markup unnoticed."""
    tags, findings = HeadTags(text), []
    if rel != "index.html" and any(href.endswith("style.css") for href in tags.hrefs("stylesheet") if href):
        findings += [f"{rel}: no preload for {font}" for font in FONTS if font not in tags.hrefs("preload")]
    if 'class="post-body"' in text:
        findings += [f"{rel}: article without {name}" for name in ("post-side", "prose") if f'class="{name}"' not in text]
    return findings


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
    posts = [SITE + path for path in listed if path not in FIXED_PAGES]
    if "rss.xml" in trees:
        findings += [f"rss.xml: item {n} has no pubDate" for n, item in enumerate(trees["rss.xml"].iter("item"), 1)
                     if not (item.findtext("pubDate") or "").strip()]
        links = [(item.findtext("link") or "").strip() for item in trees["rss.xml"].iter("item")]
        findings += site_links("rss.xml", posts, links, links)
    if (root / "llms.txt").is_file():                 # a post with a Markdown copy is linked by that copy
        urls = re.findall(r"^- \[.*?\]\((\S+?)\)", (root / "llms.txt").read_text(errors="replace"), re.M)
        findings += site_links("llms.txt", posts, urls, [re.sub(r"index\.md$", "", url) for url in urls])
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
            findings += [f"{section}/{src.stem}/index.md: broken link {url}" for url in MD_LINK.findall(MD_CODE.sub("", text))
                         if not link_ok(root, md, url)]
        for md in sorted(root.glob(f"{section}/*/index.md")):   # a renamed source leaves its old copy online
            src = root / "tools" / section / "src" / f"{md.parent.name}.md"
            if not src.is_file():
                findings.append(f"{md.relative_to(root).as_posix()}: no source {src.relative_to(root).as_posix()}")
    return findings + head_findings(root)


def site_links(name, posts, urls, pages):
    """A feed's findings: each listed post (a full URL) missing from the pages its urls name, and each url not on
    this site."""
    return ([f"{name}: missing {url}" for url in posts if url not in pages]
            + [f"{name}: link {url} is not on {SITE}" for url in urls if not url.startswith(SITE + "/")])


def book_findings(root):
    """The exception to "listings complete" (CONSTRAINTS.md), checked the other way round: each /book/ page carries
    noindex, no feed or index names the book, no list page or About links it, and the home links it once, from
    the room's door. The book's own index must still list every book page (the listing loop in check)."""
    findings = [f"{p.relative_to(root).as_posix()}: no robots noindex" for p in sorted(root.glob("book/**/*.html"))
                if not NOINDEX.search(p.read_text(errors="replace"))]
    findings += [f"{name}: names {BOOK}" for name in (*FEEDS, "search-index.json")
                 if (root / name).is_file() and BOOK in (root / name).read_text(errors="replace")]
    for rel, base, allowed in (("index.html", "/", 1), ("about.html", "/about.html", 0),
                               ("posts/index.html", "/posts/", 0), ("leetcode/index.html", "/leetcode/", 0)):
        n = sum(kind == "a" and path.removeprefix(SITE).startswith(BOOK) for kind, path, _ in links(root, rel, base))
        if n > allowed:
            findings.append(f"{rel}: links {BOOK} {n} times (allowed: {allowed})")
    return findings


def search_findings(root, listed):
    """SPEC-site-search. Active once tools/search/build.py exists: every post a listing links is in the index."""
    if not (root / "tools/search/build.py").is_file():
        return []
    try:
        urls = {e["u"] for e in json.loads((root / "search-index.json").read_text())}
    except (OSError, ValueError, KeyError, TypeError) as error:   # the error's name only, never file text
        return [f"search-index.json: unreadable ({type(error).__name__})"]
    return [f"search-index.json: missing {path}" for path in dict.fromkeys(listed) if path not in urls]


def check(root, denylist, terms_only=False):
    root = Path(root)
    deny = [(term, term_forms(term)) for term in denylist]
    findings, ids = [], {}
    # Names too, binary files included: the builds print paths and Pages serves names, so acme-corp.png leaks.
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".git" not in path.parts:
            rel = path.relative_to(root).as_posix()
            findings += [f"{rel}: private term in file name {term!r}" for term, rxs in deny
                         if any(rx.search(name) for rx in rxs for name in (unquote(rel), NAME_SEP.sub(" ", unquote(rel))))]
    for path, text in text_files(root):
        rel = path.relative_to(root).as_posix()
        # A link such as /%61cme%20corp/ is the term once served. Per line, with %0A read as a space, so a decoded
        # newline cannot shift the line numbers reported below.
        decoded = "\n".join(unquote(line).replace("\n", " ") for line in text.split("\n"))
        # Each also with file-name gaps (acme-corp quoted in a page); NAME_SEP swaps one char for one, keeping lines.
        bodies = {b for body in (text, decoded) for b in (body, NAME_SEP.sub(" ", body))}
        for term, rxs in deny:                        # whole text, so a term split across lines is still found
            lines = sorted({body.count("\n", 0, m.start()) + 1 for body in bodies for rx in rxs for m in rx.finditer(body)})
            findings += [f"{rel}:{n}: private term {term!r}" for n in lines]
        if terms_only:
            continue
        if path.suffix in PAGES or (path.suffix == ".md" and path.parent.relative_to(root).as_posix() in SOURCES):
            for n, line in enumerate(text.splitlines(), 1):
                findings += [f"{rel}:{n}: {name}" for name, rx in HTML_PATTERNS.items() if rx.search(line)]
        if path.suffix not in PAGES:
            continue
        findings += [f"{rel}: missing {name}" for name, rx in HEAD.items() if not rx.search(text)]
        urls = [a or b for a, b in LINK.findall(text)]
        findings += [f"{rel}: broken link {url}" for url in urls if not link_ok(root, path, url)]
        findings += [f"{rel}: broken fragment {url}" for url in urls if not fragment_ok(root, path, url, ids)]
        findings += shell_findings(rel, text)
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
    # Each post folder must be linked from its listing page: posts/ from the posts index, leetcode/ from its series page,
    # book/ from the book's own index (which nothing else lists: book_findings).
    for section in ("posts", "leetcode", "book"):
        linked = {path for kind, path, _ in links(root, f"{section}/index.html", f"/{section}/") if kind == "a"}
        for post in sorted(root.glob(f"{section}/*/index.html")):
            if f"/{section}/{post.parent.name}/" not in linked:
                findings.append(f"{section}/index.html: {section}/{post.parent.name}/ is not listed")
        if (root / section / "index.html").is_file():
            findings += topic_findings(f"{section}/index.html", (root / section / "index.html").read_text(errors="replace"))
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
    findings += search_findings(root, listed)
    findings += book_findings(root)
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
