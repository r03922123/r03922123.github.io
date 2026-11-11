#!/bin/sh
# The seven CONSTRAINTS.md gates for merge and push, as twelve named steps; this is the only place they are spelled out.
# The same script runs locally before every push and in CI (.github/workflows/gates.yml), so the two cannot drift.
# One named step per gate, so a red log says which rule broke. Privacy runs first and stops the run,
# so no later step can echo a private term out of a file. The quick steps run before the slow builds.
set -u
cd "$(dirname "$0")/.." || exit 1

gate() { name=$1; shift; echo "== $name"; "$@" || { echo "FAILED: $name"; exit 1; }; }

check_tests() { (cd scripts && python3 -m unittest test_check); }
voice_tests() { (cd scripts && python3 -m unittest test_voice_check); }

# One allowed hit, the InterviewQuery credit. -l and -c print names and counts, never a matching line.
WORDING='interview|obsidian|vault|musicgen'
ALLOWED=tools/posts/src/coin-flips-normal-approximation.md
post_wording() {
  files=$(grep -l -i -E "$WORDING" tools/posts/src/*.md)
  [ "$files" = "$ALLOWED" ] || { echo "files with a match: $files (allowed: $ALLOWED only)"; return 1; }
  count=$(grep -c -i -E "$WORDING" "$ALLOWED")
  [ "$count" -eq 1 ] || { echo "$ALLOWED has $count matching lines (allowed: 1)"; return 1; }
  grep -i -E "$WORDING" "$ALLOWED" | grep -q InterviewQuery || { echo "the one match is not the InterviewQuery credit"; return 1; }
}

# Generated HTML is committed: after the builds the tree must equal HEAD, with no untracked files either.
rebuild_clean() {
  changes=$(git status --short) || { echo "git status failed"; return 1; }   # an error must not read as clean
  [ -z "$changes" ] || { echo "$changes"; return 1; }
}

gate "privacy"        python3 scripts/check.py --denylist-only
gate "site"           python3 scripts/check.py
gate "check tests"    check_tests
gate "search tests"   node --test scripts/test_search.mjs   # ranking cases and golden queries, on the committed index
gate "post wording"   post_wording
gate "voice tests"    voice_tests                               # post-voice only: PR 8 removes these two
gate "post voice"     python3 scripts/voice_check.py            # code, tables and numbers match voice-baseline.json
gate "post build"     python3 -I tools/posts/build.py .
gate "leetcode build" python3 -I tools/leetcode/build.py .
gate "feeds build"    python3 -I tools/feeds/build.py .         # reads both post lists, so it runs after them
gate "search index"   python3 -I tools/search/build.py .        # reads the posts; rebuild clean then covers it
gate "rebuild clean"  rebuild_clean
echo "all gates passed"
