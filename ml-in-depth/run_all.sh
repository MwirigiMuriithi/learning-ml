#!/usr/bin/env bash
# Regenerate every GIF by running every lesson script. Usage: bash run_all.sh [lesson-prefix ...]
set -u
cd "$(dirname "$0")"
want=("$@")
for d in lessons/*/; do
  name=$(basename "$d")
  if [ ${#want[@]} -gt 0 ]; then
    ok=0; for p in "${want[@]}"; do [[ $name == $p* ]] && ok=1; done
    [ $ok -eq 0 ] && continue
  fi
  for f in "$d"*.py; do
    [ "$(basename "$f")" = "cart.py" ] && continue            # helper module, not a lesson script
    echo "=== $f"
    ( cd "$d" && python3 "$(basename "$f")" ) || echo "!!! FAILED: $f"
  done
done
