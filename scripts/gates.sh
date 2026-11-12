#!/bin/sh
# The seven CONSTRAINTS.md gates for merge and push, as twelve named steps; this is the only place they are spelled out.
# The same script runs locally before every push and in CI (.github/workflows/gates.yml), so the two cannot drift.
# One named step per gate, so a red log says which rule broke. Privacy runs first and stops the run,
# so no later step can echo a private term out of a file. The quick steps run before the slow builds.
set -u
cd "$(dirname "$0")/.." || exit 1

gate() { name=$1; shift; echo "== $name"; "$@" || { echo "FAILED: $name"; exit 1; }; }

check_tests() { (cd scripts && python3 -m unittest test_check); }

# A post or book source never names where its notes came from. -l prints names, never a matching line.
no_source_names() {
  files=$(grep -l -i -E 'obsidian|vault' "$@"); status=$?
  [ "$status" -le 1 ] || { echo "grep failed ($status)"; return 1; }    # 1 is "no match"; 2 is an error, not a pass
  [ -z "$files" ] || { echo "files with a match: $files"; return 1; }
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
gate "post wording"   no_source_names tools/posts/src/*.md
gate "book wording"   no_source_names tools/book/src/*.md
gate "post build"     python3 -I tools/posts/build.py .
gate "leetcode build" python3 -I tools/leetcode/build.py .
gate "book build"     python3 -I tools/book/build.py .          # unlisted: the feed and search builds skip it
gate "feeds build"    python3 -I tools/feeds/build.py .         # reads both post lists, so it runs after them
gate "search index"   python3 -I tools/search/build.py .        # reads the posts; rebuild clean then covers it
gate "rebuild clean"  rebuild_clean
echo "all gates passed"
