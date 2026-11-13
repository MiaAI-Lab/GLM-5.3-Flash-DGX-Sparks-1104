#!/usr/bin/env bash
# Second half of the upstream decode A/B (after ab-upstream.sh's base and l2pf legs): the copy bench on the serving
# l2pf image, then patch 0115 (32-row windows) in, and its legs with L2PF on and off, each with the decode sweep,
# the token-hash contract and the copy bench.
#   ab-upstream2.sh <path-to-0115.patch>
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
P115=${1:?usage: ab-upstream2.sh <0115 patch>}
LOG=docs/speed/ab-upstream.log
say() { printf '%s\n' "$*" | tee -a "$LOG"; }
say ""; say "########## copy bench on the serving image (l2pf)  $(date -u +%H:%M:%SZ) ##########"
python3 -I docs/speed/copybench.py up-l2pf 2>&1 | tee -a "$LOG"
cp "$P115" patches/engine/ || { say "  cannot place 0115"; exit 3; }
for leg in w32l2pf=L2PF=1 w32=L2PF=0; do
  label=${leg%%=*}; vars=${leg#*=}
  say ""; say "########## $label ($vars)  $(date -u +%Y-%m-%dT%H:%M:%SZ)  patches=$(ls patches/engine/*.patch | wc -l) ##########"
  ( export "$vars"; bash docs/safestart.sh ) >> "$LOG" 2>&1 || { say "  FAILED to start $label"; exit 3; }
  docker ps --filter name=glm53-spark --format '  image={{.Image}}' | tee -a "$LOG"
  LOADS=1,4,8 TRIALS=3 python3 -I docs/dec.py "up-$label" 2>&1 | tee -a "$LOG"
  python3 -I docs/dec.py "up-$label" --contract 2>&1 | tee -a "$LOG"
  python3 -I docs/speed/copybench.py "up-$label" 2>&1 | tee -a "$LOG"
  say "########## $label done  $(date -u +%H:%M:%SZ) ##########"
done
