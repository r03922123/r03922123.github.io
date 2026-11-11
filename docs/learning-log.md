# Learning log

The three most important whys of each change, newest first. Each heading reads `<date> · <phase>: <title>`.

## 2026-11-11 · build: post-voice PR 5, probability

1. **A test is not a proof, so the limit says which one it is.** The probability posts check each formula at a few
   parameter values. That catches a wrong formula or a mixed-up convention, but it does not prove the formula for
   every input, and the cheat sheet's new limit says so. It is like weighing three loaves to check a scale: a broken
   scale shows, a good one is not certified. Trade-off: one more caveat on a page meant to be quick. Risk: a reader
   may trust the formulas less than they should; the linked posts carry the derivations.
2. **An open question names its next tool.** Each closing paragraph says what the post's model cannot answer and
   what to reach for: the beta-binomial for correlated trials, Weibull or Kaplan-Meier when the hazard is not
   constant, a zero-inflated model for excess zeros. Each tool comes from the post's own pitfalls. Trade-off: the
   ending points away from the post. Risk: a named tool can look like an endorsement, so it must already be in the
   post.
3. **Some lines are load-bearing.** `coin-flips-normal-approximation` holds the one credit the `post wording` gate
   allows, so its drafter was told to leave that line exactly as it was. A rewrite has to respect every gate, not
   only the voice check. Trade-off: one line in one post is off limits. Risk: a gate that greps for a word can be
   tripped by an innocent sentence, which is why the wording gate runs before every push.

## 2026-11-11 · build: post-voice PR 4, statistics and experimentation

1. **Change the voice, not the claim.** `causal-inference` had an overstated sentence written in "we". The spec
   forbids "we" and also forbids hiding a correction, so only the pronouns changed, which made it passive ("where the
   true answer is known") and the overstatement went on the findings list. It is like correcting a typo in a quote
   without changing what the speaker said. Trade-off: the sentence still overstates until its own fix. Risk: an
   impersonal rewrite can drift in meaning, so the reviewer compares it with the original.
2. **Say what a simulation builds in.** A test that plants an assumption and then recovers it shows the method
   works, not that the assumption holds. So the new limits say so outright: "each simulation here is built so that
   its method's assumption holds, except the DiD run that breaks it on purpose". Trade-off: the posts sound less
   sure of themselves. Risk: a reader may take the limit as doubt about the maths, which the asserts do settle.
3. **Put the assert into words.** An assert such as `abs(wald - 2) < 0.15` proves a result, but few readers parse it.
   One sentence after it ("the Wald ratio lands within 0.15 of the compliers' 2") makes the evidence readable.
   Trade-off: a few words of the 10% growth budget. Risk: the sentence can round the assert into a stronger claim, so
   each one quotes the bound the code checks.

## 2026-11-11 · build: post-voice PR 3, machine learning fundamentals

1. **An overstated sentence stays as it was, and goes on a list.** Six existing sentences in six posts claim more
   than their code shows, such as "checks each claim in code" and "each with a small experiment". Turning those into
   "I check …" would make the overstatement the author's own, and quietly fixing them would hide a correction in a
   style diff. So they stay word for word, and the PR lists them for their own fix, like an erratum filed apart from
   the next edition. Trade-off: known soft spots stay live until that fix. Risk: the list lives in a PR description,
   so the next session must carry it forward.
2. **Name the kind of evidence.** The new limits say which claims the code measures and which it only argues or
   cites: "argued, not measured", "cited, not reproduced", "I train no deep network". A reader can then weigh each
   claim. Trade-off: more caveats in a post that is already careful. Risk: a limit can undersell a claim the code
   does check, so the reviewer reads each one against the code.
3. **Draft in parallel, check in series.** Three drafters each took a few posts in their own scratch copy. Then the
   author read every new sentence, and a fresh reviewer checked them against the code, like three proofreaders where
   only the last one signs off. Trade-off: two readings per post. Risk: drafters share one brief, so a blind spot in
   the brief repeats across 11 posts; the brief now says how to handle an overstated sentence.

## 2026-11-11 · build: post-voice PR 2, the 3-post pilot

1. **Pilot on three kinds of post first.** `dropout` is a mechanism, `ab-testing` an experiment and `position-bias` a
   long-form page, so the voice gets tried on each shape before 56 more posts copy it. A tailor makes a toile before
   cutting the good cloth. Trade-off: the owner reads the pilot before PR 3, which holds up the rollout. Risk: if a
   fourth kind of post (a cheat sheet) needs something different, the pilot can't show it; the spec gives those
   only the opening and limits moves.
2. **Lead with a number the post already proves.** Each new opening gives a result from the post's own table,
   simulation or caption, such as `dropout`'s training MSE of 0.03 on noisy points against a test MSE of 0.17 on the
   true curve. The definition comes second. Trade-off: the opening is denser. Risk: a number lifted from its table
   can lose its conditions, so each one gets its limit nearby ("one sine-wave task averaged over four seeds").
3. **Name the limits from what the code does.** Each closing paragraph lists what the post's code assumes (the
   simulation knows the true propensities; users are independent and the effect stays constant). It doesn't list
   generic caveats. Trade-off: the 10% growth cap leaves room for about one paragraph. Risk: a limit can
   still be framed wrong, which only a reader catches, so the reviewer checks each one against the code.

## 2026-11-11 · build: post-voice PR 1, the voice check and its baseline

1. **Freeze the evidence before touching the words.** `voice_check.py` records each post's code blocks, tables,
   inline code, maths, headings, links and prose numbers in `voice-baseline.json`, taken from `main` before any
   edit. Like photographing a crime scene before anyone walks in: later PRs can reword freely, and the check shows
   nothing else moved. Trade-off: a 78 KB generated file lives in the repo until PR 8 deletes it. Risk: if someone
   regenerates the baseline after an edit, the check would bless that edit, so it refuses to overwrite itself.
2. **Compare numbers, spans and links as sets, not text.** A rewrite may move "17,000 weights" into another sentence,
   say "4 seeds" because the code says `seeds = 4`, or reuse \(p\) in a sentence about a table. So a prose number
   passes if the post already had it in its prose, code, tables, inline code, maths, headings or reference list, and
   fails if it is new or gone; inline code, maths and links follow the same rule. Trade-off: repeating one, or
   dropping one of two mentions, is not caught, nor are "half", "twice", a lost minus sign or a comma list like
   "64,128,256", which reads as one number. Markdown credit lines are prose too: only their links and numbers are
   guarded. Risk: the check can't tell whether a number still means the same thing, which is why a fresh reviewer
   reads every new sentence.
3. **Parse the way the build does.** The check splits a source with the same line rules as `tools/leetcode/build.py`
   (fences, `|` tables, `\[` blocks, `##` headings), so it sees the post the reader sees. On the 4 long-form pages it
   leaves out only `reading-shell`'s own header, nav and aside, by class, so a new callout counts as prose.
   Trade-off: if the Markdown engine gains a new block type, the check must learn it too. Risk: a block it doesn't
   know counts as prose, so only the numbers in it stay guarded.

## 2026-11-11 · build: expert accuracy pass over all 90 posts

1. **Fix facts apart from style.** This pass changed only claims that a specialist in each field would mark wrong or
   out of date: 14 edits in 13 of the 90 posts. Wording, structure and voice wait for `post-voice`. It is like a paper's
   errata versus its next draft: an erratum lists one fix per claim, so a reader can check each one. Trade-off: two
   passes over the same posts instead of one. Risk: `post-voice` later rewrites a fixed sentence and brings the old
   claim back, so its reviewer should read this PR's list.
2. **Split reviewers by field, not by file count.** Ten reviewers each took one field (statistics, LLMs, recommender
   systems, algorithms and so on), so each read posts it could judge. Like peer review, a paper goes to someone who
   knows its area. Trade-off: the groups are uneven, from 6 to 16 posts. Risk: a post spanning two fields was
   checked from one side only.
3. **Leave what you cannot verify.** Reviewers left 37 doubts unedited because they could not confirm them
   from what they knew, among them 2026 paper details and whether a library is still maintained. A wrong
   "correction" on a public page is worse than a cautious sentence. Trade-off: some soft spots stay live. Risk: those
   doubts live only in the PR description, so nothing reminds anyone to come back to them.

## 2026-11-11 · build: site-search

A search box on the Posts and LeetCode lists, over all 92 entries, from a 65.5 KB index (21.7 KB gzipped).

1. **Index the labels, not the books.** The index holds titles, summaries, tags, sections and headings, not full
   text, which would cost 25 times as much. It is like a library catalogue: you find the book by its card, then open
   it. Trade-off: a word that appears only in a post's body is not found. Risk: a post whose title and headings never
   name its key term cannot be reached by that term; the 13 golden queries guard the cases we know about.
2. **Keep the ranking pure and test it like a function.** `rank()` takes entries and a query and touches no page, so
   `node --test` checks its rules (exact beats prefix, title beats summary, every word required, stable ties) and the
   golden queries on the real index. It is like validating a model offline on a fixed test set before shipping it.
   Trade-off: the box itself (focus, URL, failure message) is checked only in a browser. Risk: the golden queries
   were picked by the designer, so passing them proves no regression, not good quality.
3. **Add it so that losing it costs nothing.** The script inserts the box, fetches the index only when the box is
   first used, and removes the box if the index fails, so without JavaScript or the index the lists work as before.
   It is like a lab's optional add-on instrument: unplug it and the main bench still runs. Trade-off: no box at all
   without JavaScript. Risk: query and index text reach the page by `textContent` only; one `innerHTML` added later
   would let a crafted `?q=` link inject markup.

## 2026-11-11 · build: posts-index

Topic cards, an article count and a topic filter on the Posts (9 cards, 61 articles) and LeetCode (6, 31) lists.

1. **Let the URL hold the filter.** A card is a plain link to `#probability`; CSS hides every other section while
   one is the URL's target. There is no JavaScript and no saved state, so a shared link opens filtered and the back
   button undoes a filter for free. It is like a bookmark in a binder: the tab you open to is the state. Trade-off:
   the filter needs `:has`, and only one topic can be chosen at a time. Risk: an id that changes breaks the shared
   links and the post pages' topic links; the fragment check catches it.
2. **Fail safe, not blank.** The rule only hides sections when one of them is targeted, so a browser without
   `:has` drops the rule and shows every section, and each card still jumps to its own. It is like a lab door that
   unlocks when the power fails. Trade-off: those browsers get a jump, not a filter. Risk: a later rule that hides
   sections by default would turn the same failure into an empty page.
3. **Check the numbers the page states.** Each card says how many posts its topic holds, so the site check now
   counts each section's rows and fails on a section with no card or a card whose count differs. It is like
   checking that a table's row totals add up before it goes in the paper. Trade-off: it reads the markup the two
   templates write, not every possible layout. Risk: it proves the count, not that the filter looks right; that is
   the browser check's job.

## 2026-11-11 · build: reading-shell layout

Second of the spec's two PRs: the layout itself, on 90 posts, two lists, About and 404.

1. **Let code grow toward the empty side.** The side column holds the right of the text and the left was empty,
   so code and tables of 5 or more columns grow left only, ending at the text's right edge, and only as far as their
   longest line needs (up to 1040 px). It is like a wide figure in a book taking the free margin, never the one with
   the notes. Trade-off: code blocks of different lengths start at different left edges. Risk: the room on the left
   is computed with a container unit (`100cqw`); a browser without it would let a long block run off the page.
2. **Switch the layout on by what a page holds, not by a list of pages.** `body:has(.post-body)` widens any page
   that holds an article, so every post gets the grid and the home, lists, About and 404 keep 760 px with no
   per-page class. It is like a lab rule for "any sample that contains X" instead of a list of sample ids: a new
   post is covered without anyone remembering it. Trade-off: it needs `:has`, which Firefox gained only in late 2023.
   Risk: a future page that reuses `.post-body` for something else gets the wide grid too.
3. **Check the pages that skip the template.** Two templates regenerate 86 posts, but 4 long-form pages, About and
   404 are edited by hand and could quietly keep the old markup. The site check now fails on a reading page without
   the two font preloads, or an article without its side and text columns. It is like an extra checklist line for
   the samples prepared by hand, outside the automated pipeline. Trade-off: it reads class names as plain strings.
   Risk: it proves the columns exist, not how they look; that is still the browser check's job.

## 2026-11-11 · build: reading-shell

First of the spec's two PRs: the fragment-link check alone, before the layout moves any heading.

1. **Guard the doors before moving the walls.** The new layout rewrites every article's header and About's frames,
   so an id could vanish and its links (a contents entry, a room portrait) would quietly land at the top of the
   page. The check comes first, while all 708 local fragment links resolve, so any break in the shell PR shows up
   as a red line. It is like taking a baseline measurement before changing the experiment. Trade-off: a separate PR
   to review. Risk: the check reads ids from the HTML file, so an id that only JavaScript adds would be reported.
2. **Each red test must fail for its own reason.** A link to another page, to a folder page and to the same page
   take three different paths through the code, and same-page links (703 of the 708) skipped the old link check
   entirely. One test per path, each seen failing first, is like a control for each condition instead of one shared
   control. Trade-off: three tests where one would look tidier. Risk: a fourth link shape, such as `?q#x`, has no
   test of its own.
3. **Reuse the resolver, do not copy it.** `link_ok` already knew how a link becomes a file (case-exact, never
   above the root), so that part moved into `target()` and both checks share it, like one calibrated instrument used
   for two measurements. Trade-off: touching a working function. Risk: a change to `target()` now moves both checks
   at once, which is why the 78 older link and page tests still run beside the new ones.

## 2026-11-11 · build: blog-theme

1. **Swap the values, keep the names.** Every rule asks for `var(--bg)` or `var(--font-body)`, so the reading pages
   change by redefining about a dozen tokens at `:root`, and the room keeps its look by redefining them again on
   `body.home`. It is like a config file per experiment: the code stays the same and only the settings differ.
   Trade-off: the two surfaces share token names, so a new token must be set in both places. Risk: a token set only
   at `:root` leaks the reading value into the room, which is why a test pins all 22 of the room's colours.
2. **A gate must follow the thing it guards.** The contrast test used to read only the `:root` blocks, so moving the
   room's colours under `body.home` would have taken them out of the test without a single failure. It now reads all
   four palettes and fails if any block is missing, like a lab checklist that lists every sample by name instead of
   "the samples in fridge A". Trade-off: the test finds blocks by their exact opening lines, so reformatting them
   means editing the test. Risk: a fifth palette added later would be missed until someone adds it to `PALETTES`.
3. **Pay for a font only where it shows.** The three font files live in the repo, and a browser downloads a face only
   when some text on the page uses it. The home names neither new font, so it requests nothing new, and the italic
   loads only on pages with italic text. It is like a library ordering a book only when a reader asks for it.
   Trade-off: until `reading-shell` adds preload tags, a reading page may draw in the fallback font for a moment
   (`font-display: swap`). Risk: five characters outside the Latin subset fall back to the next font in the stack.

## 2026-11-11 · build: post-feeds head tags

1. **One function writes the tags.** `head_tags()` in `tools/leetcode/build.py` makes the canonical link, feed link,
   JSON-LD and Markdown alternate for both builds, and its output was pasted into the six hand-written pages. It is
   like a rubber stamp instead of hand-lettering 94 labels. Trade-off: the six hand-written pages do not follow a
   later change to the function. Risk: they drift, and `check.py` sees only what it checks: a missing or wrong URL,
   JSON that does not parse, an email, but not a renamed author.
2. **A canonical link names the original.** The same post answers at `/posts/x/` and `/posts/x/index.html`; the
   canonical link tells a crawler which one to index, like a DOI that names the paper among its mirrors. Trade-off:
   one more line per page. Risk: a canonical that points at another page quietly drops this one from search, so the
   check requires exactly one, equal to the page's own URL.
3. **Two renderings of one source must agree.** The page drops a link to an unwritten post, but `index.md` is the
   raw source, so the link stayed live there. `check.py` now runs the page's link check on every Markdown link
   outside code, like proofreading the PDF and the HTML version of a paper, not just one. Trade-off: the build does
   not unlink, so such a link now fails the gate instead of waiting for its post. Risk: a regex is not a Markdown
   parser, so an unusual link shape could slip past.

## 2026-11-11 · build: post-feeds feed files

1. **Same input, same bytes.** The feed build reads no clock: dates come from the posts' fixed dates, and ties keep
   the listing's order. It is like a seeded experiment, where rerunning gives the identical table. Trade-off: no
   `lastBuildDate` or `lastmod`, which feed readers and crawlers would like. Risk: a "now" added later would make
   every rebuild a change, and the rebuild gate would fail on every run.
2. **Leave out the field instead of filling it.** RSS's `author` must be an email, and the site's feeds carry none,
   so the field is omitted and `check.py` fails on any email in a feed file. It is like dropping a column that
   would identify a patient instead of hoping no one reads it. Trade-off: feed readers show no author name. Risk:
   the email pattern skips `name@1.2.3`, so a CDN pin is not a finding, and an address with an unusual shape could
   slip past.
3. **A check must fail when its subject is gone.** The feed rules switch on once `tools/feeds/build.py` exists and
   then require every output, like a lab check that flags a missing sample instead of skipping that row. Trade-off:
   the rule depends on a file existing, which a test pins down. Risk: renaming the build file switches the rules off
   quietly, so a rename must update `check.py` too.

## 2026-11-11 · understand: CI hardening from the site-baseline review

1. **Give a pattern one way to match.** The scrubber read `< >` as one tag or as three characters, so each such gap
   doubled the work, like a maze where every fork leads to the same room. A `<...>` now counts as a tag only if
   it holds a letter or digit that no other branch can read, so `< >`, `<%20>` and `<&nbsp;>` are read piece by
   piece, and each gap has one reading. Trade-off: a harder pattern, split into named parts. Risk: three quicker
   fixes each fell short. A possessive repeat missed `%20`, "a tag starts with a letter" missed `<1>` (a punctuated
   term printed in clear text), and "a tag holds a letter" left `<&nbsp;>` slow. Tests and two fresh reviews
   caught them. The final rule accepts the same gaps as the old one on 111,150 test strings.
2. **Measure the slow case in a test.** The new test times crafted lines: seconds to minutes before (90 s for 24
   `<%20>` gaps on one machine), milliseconds after.
   Trade-off: a timing test can flake on a slow machine, so it allows 1 s for a case that takes milliseconds.
   Risk: the bound is loose enough that a smaller slowdown would pass.
3. **Cancel stale PR runs, never `main`'s.** A newer push to a PR makes the older run pointless. Each commit on
   `main` is live, though, so each one gets its own concurrency group and finishes its own check. One shared
   group was not enough: GitHub keeps one waiting run per group and cancels it when a third push arrives.
   Trade-off: quick merges run in parallel and cost more minutes. Risk: none beyond those minutes.

## 2026-11-10 · stop the bleeding: site-baseline (gates script, CI, protected main)

1. **Fail closed.** A check that cannot see its input should say no, not yes, like a smoke alarm that beeps when its
   battery is gone. `check.py` now exits non-zero when the denylist has no terms, so an unset CI secret turns the
   run red. Trade-off: a PR from a fork, which gets no secrets, always fails. Risk: someone "fixes" that by
   allowing an empty list, which would make the privacy gate pass on nothing.
2. **Scrub at the one exit, not at each print.** Every line `check.py` prints, tracebacks included, goes through
   one scrubber, like one security gate at an airport instead of a guard at every shop. A term shows as
   `[term #3]`. Trade-off: local debugging needs `--show-terms`. Risk: commands outside `check.py` are not
   scrubbed, so privacy runs first and stops the run, and the wording gate prints names and counts, never lines.
   Hiding too much leaks as well: a label inside `planning` would mark `ann` as private, so the scrubber hides only
   whole words, with the same word gaps that detection uses. Making those gaps match found a term already on `main`.
3. **The same script locally and in CI.** `scripts/gates.sh` is the only place the gates are written, like one
   recipe card used at home and in the restaurant. Trade-off: CI must install the post code's exact packages
   (`requirements-ci.txt`, every package locked by hash, because post code runs while the denylist is on disk). Risk: numbers can differ on Linux; if a post fails only there, stop and report it
   rather than loosen an assert.

## 2026-11-10 · stop the bleeding: capability map v4 and seven module specs (PR #2)

1. **Specs merge before any build.** Each build session starts from one written source instead of a chat.
   Trade-off: a large docs PR. Risk: specs drift from the code, so each spec lists the code facts it relies on.
2. **Freeze the evidence, then reword.** Like sealing lab results before editing the paper's prose: `post-voice`
   checks by hash that code, tables and numbers stay the same while the framing changes. Trade-off: a temporary
   check and a baseline file. Risk: a new sentence can still be untrue, so a fresh reviewer reads every one.
3. **Measure our own content before copying a layout.** Like checking a recipe against what is in your kitchen:
   530 of 585 code blocks are wider than the reference's 720 px column, so code gets a wider column. Trade-off: the
   design departs from the reference where the data differs. Risk: a few measures rest on small samples; the specs
   say which.
