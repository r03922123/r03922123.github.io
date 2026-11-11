# Constraints

Last reviewed: 2026-11-01 by the owner (r03922123), written up by Claude. Gates moved into `scripts/gates.sh` and CI
on 2026-11-10 (`SPEC-site-baseline.md`).

This repo is a **public** GitHub Pages site: whatever reaches `main` on GitHub is live. These rules decide when
work gets saved, checked, merged and published. Do not weaken this file to make a change pass. Tightening a rule
can be a quiet change; loosening one must be said in the commit message.

## Workflow

| Rule | Why |
|---|---|
| Feature work goes on a branch, never straight on `main`. | `main` is what Pages serves; a branch keeps unfinished work off the live site and makes each change reviewable as a unit. |
| Commit after each finished unit: one post, one fix, one spec change. | Commits are save points. 11 h of uncommitted v2 work (2026-11-01) could not be reviewed, bisected or reverted in pieces. |
| No more than **1 h of uncommitted work** at any time. | Caps what one crash, bad edit or stray `git checkout` can lose. The Stop hook below warns when it looks exceeded. |
| Checks gate **merge and push**, not local commits. | A local commit is a private save point and should be cheap. What must be clean is what leaves the machine. A failing check is fixed in a later commit on the branch, before merge. |
| Push and merge only with the owner's explicit OK. | Publishing is public and hard to take back (pages get cached and indexed). |

## Gates for merge and push

`sh scripts/gates.sh` runs them all, locally before every push and in CI as the required `gates` check on every PR
to `main`. Each row names its step in that script, the only place the commands are spelled out. All must pass on the
branch tip before it is merged or pushed.

| Gate | Rule | `gates.sh` step |
|---|---|---|
| Builds | Every post, book and LeetCode page builds, and every python block passes its asserts; the feed files and the search index build | `post build`, `leetcode build`, `book build`, `feeds build`, `search index` |
| Privacy and site | No private terms (local denylist) in contents or file names, no phone numbers, no broken links, page heads complete, pinned CDNs, listings complete | `privacy`, then `site` (needs `scripts/denylist.local.txt`; in CI, the `SITE_DENYLIST` secret) |
| Home weight | `index.html` + `style.css` ≤ 150 KB (three.js lazy and counted apart) | `site` |
| Check tests | The checker itself still works; the search ranking passes its cases and golden queries | `check tests`, `search tests` |
| Post wording | No interview framing, vault or Obsidian mentions, no MusicGen authorship; the one allowed hit is the InterviewQuery credit in `coin-flips-normal-approximation.md` | `post wording` |
| Book wording | `tools/book/src` may use interview framing (ADR 0003) but never names the notes' source | `book wording` |
| Rebuild is clean | Generated HTML is committed and matches its sources | `rebuild clean` |

## Measured, not yet enforced

| Metric | Today (2026-11-01) | Direction | Source |
|---|---|---|---|
| Home first contentful paint, warm GPU | 156 ms phone, 228 ms desktop | must not grow | J4 audit, headless Chrome over CDP, localhost |
| Home draw calls per frame | 157 measured before instancing the bands; about 113 estimated after | must not grow | J4 audit; re-measure |
| Room frame rate | 30 fps active, 15 idle or touch | fixed by design (SPEC-room) | J4 audit + `room.js` |

None of these has an external check yet (Lighthouse is not installed). CI runs the gates above but not these
metrics; `.github/workflows/gates.yml` is the natural place for them later.

## Exceptions

- `/book/` (ADR 0003): unlisted on purpose, so it is exempt from "listings complete"; `site` checks the
  opposite instead, that no listing, feed or index names it. Every other page keeps the full rule.
