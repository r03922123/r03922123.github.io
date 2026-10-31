#!/bin/sh
# shoot.sh <name> <light|dark> <desktop|phone> <target> : target is a site path (served on :8765), or a file:// or http URL
S="${SCRATCH:-$(cd "$(dirname "$0")/../.." && pwd)/.cache}"; mkdir -p "$S/shots"
C="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
scheme=1; [ "$2" = dark ] && scheme=0                      # Blink: 0 = dark, 1 = light
case "$4" in file://*|http*) url="$4" ;; *) url="http://localhost:8765/$4" ;; esac
if [ "$3" = phone ]; then size=600,1200; url="file://$(cd "$(dirname "$0")" && pwd)/frame.html#$(python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1],safe=""))' "$url")"; else size=1440,1000; fi
"$C" --headless=new --disable-gpu --hide-scrollbars --user-data-dir="$S/chrome-$1" --blink-settings=preferredColorScheme=$scheme \
  --window-size=$size --virtual-time-budget=4000 --screenshot="$S/shots/$1.png" "$url" >/dev/null 2>&1
echo "$S/shots/$1.png"
