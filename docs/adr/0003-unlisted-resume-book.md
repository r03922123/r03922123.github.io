# ADR 0003: Publish résumé-preparation notes, unlisted, behind a book in the room

Status: accepted, 2026-11-11 (owner, in chat). Recorded by `resume-book`.

## Context

The owner's private notes hold résumé-preparation material: one note per résumé project, written for telling that
project well. Two rules kept it off the site: `SPEC-posts-vault.md` says never to publish interview coaching or
résumé details beyond About, and the `post wording` gate fails on interview framing in post sources. On 2026-11-11
the owner asked to publish this material, reachable only by clicking a hidden purple book in the common room.

## Decision

Publish rewritten résumé-preparation pages under `/book/`, built from `tools/book/src/` by `tools/book/build.py`.
The two rules get scoped exceptions, not deletions:
- `SPEC-posts-vault.md`'s "never publish interview coaching or résumé details" still holds for `/posts/`. It does
  not apply to `/book/`, which `SPEC-resume-book.md` governs.
- The `post wording` gate still scans `tools/posts/src/` unchanged. `tools/book/src/` gets its own `book wording`
  gate, which allows interview framing and still forbids naming the notes' source.

Personal identifiers stay out everywhere: people's names, recruiter and HR contacts, companies tied to job
applications, compensation, contact details, application dates and statuses, health and family matters.

## Hidden is not private

The book hides the pages from casual visitors, not from anyone who looks. The door is an `<a href="book/">` in
`index.html`, readable in the page source and by any crawler; the URL is short and guessable; and the repo is
public, so the Markdown sources sit in `tools/book/src/` on GitHub. What the site does:
- each page carries `<meta name="robots" content="noindex">`, which search engines that honour it obey;
- the pages are absent from the sitemap, RSS, `llms.txt`, the search index and the post lists, and `check.py`
  fails if one appears there;
- `robots.txt` does not disallow `/book/`: a crawler must fetch a page to see its `noindex`, and a `Disallow` line
  would also announce the path.

So the pages must be fit for anyone to read. Only content the owner would show a stranger goes in the book.

## Consequences

- Gives up the old line between public posts and private preparation notes; the line moves to "no personal
  identifiers", which the denylist, `check.py` and review enforce.
- A new source folder and build, so the feed and search builds stay as they were, and one more `gates.sh` step.
- Pages may be cached or archived by third parties despite `noindex`; taking one down later does not erase copies.

## Alternatives dropped

- Pages under `/posts/` with an exclusion list in the feed and search builds: every build and check would need an
  exception, and a missed one would list the page.
- A private repo or password page: GitHub Pages cannot gate access, and the owner wants the pages public.

## Update 2026-11-12

`post wording` now fails only on obsidian or vault (PR #37). Keeping interview framing out of `/posts/` is a
review rule in `SPEC-posts-vault.md`, no longer a gate.
