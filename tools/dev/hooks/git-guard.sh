#!/bin/sh
# Stop hook (CONSTRAINTS.md, Workflow): warn, never block, when uncommitted work piles up or sits on main.
# Prints {"systemMessage": ...} for Claude Code to show the owner; prints nothing when all is well. Always exits 0.
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" 2>/dev/null || exit 0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0
n=$(git status --porcelain | wc -l | tr -d ' ')
branch=$(git branch --show-current)
msg=""
if [ "$n" -gt 0 ] && [ "$branch" = "main" ]; then
  msg="$n uncommitted change(s) on main. CONSTRAINTS.md: feature work goes on a branch (git switch -c <name>)."
fi
if [ "$n" -gt 15 ]; then
  msg="${msg:+$msg }$n changed or untracked entries in git status (more than 15). CONSTRAINTS.md: commit after each finished unit; at most 1 h of uncommitted work."
fi
[ -n "$msg" ] && printf '{"systemMessage": "git guard: %s"}\n' "$msg"
exit 0
