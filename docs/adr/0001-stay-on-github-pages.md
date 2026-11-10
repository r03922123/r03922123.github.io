# ADR 0001: Stay on GitHub Pages, with the build run locally

Status: accepted, 2026-11-10 (owner). Recorded by `site-baseline`; first stated in `CAPABILITY-MAP.md`, version 3.

## Context

The site is static HTML, built from Markdown by `tools/posts/build.py` and `tools/leetcode/build.py`, and the
generated HTML is committed. Pages serves `main` as it is. The redesign takes its structure from claude.dev, which
runs on a framework and host of its own.

## Decision

Keep GitHub Pages and the Python build. Adopt only claude.dev's public outputs (layout, feeds, search), not its
framework. CI checks every PR with `scripts/gates.sh`; it does not deploy. Deploying means merging a PR into `main`.

## Consequences

- Gives up custom response headers: Pages cannot set them, so security headers such as a Content-Security-Policy
  are out of reach.
- Generated HTML stays in git, so the `rebuild clean` gate is needed to catch a page edited by hand.
- Every file in the repo is served (`.nojekyll`), so project docs must hold nothing private.
- Kept: no hosting cost, no new account, and the site keeps working if CI is down.

## Alternatives dropped

- Another host with headers and builds on deploy: a new account and a migration, for headers this site does not
  need yet.
- Deploying to Pages from Actions: it would remove the committed HTML, but it is a separate change with its own
  risks, and nothing in the redesign needs it.
