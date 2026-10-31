#!/bin/sh
# shoot3d.sh <name> <desktop|phone> <url> : WebGL-capable headless screenshot (SwiftShader); prints the PNG path.
# Chrome hangs after writing the screenshot, so a hard 75 s alarm ends it; the PNG is already on disk by then.
S="${SCRATCH:-$(cd "$(dirname "$0")/../.." && pwd)/.cache}"; mkdir -p "$S/shots"
C="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
url="$3"
if [ "$2" = phone ]; then size=600,1000; url="file://$(cd "$(dirname "$0")" && pwd)/frame.html#$(python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1],safe=""))' "$url")"; else size=1440,900; fi
rm -f "$S/shots/$1.png"
perl -e 'alarm 75; exec @ARGV' "$C" --headless=new --hide-scrollbars --user-data-dir="$S/chrome3d-$1" --use-angle=swiftshader --enable-unsafe-swiftshader \
  --window-size=$size --timeout=20000 --screenshot="$S/shots/$1.png" "$url" >/dev/null 2>&1
[ -f "$S/shots/$1.png" ] && echo "$S/shots/$1.png" || echo "NO SCREENSHOT for $1"
