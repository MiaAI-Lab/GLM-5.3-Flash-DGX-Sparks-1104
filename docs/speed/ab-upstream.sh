#!/usr/bin/env bash
# A/B legs for the upstream decode ports (0113 L2PF, 1779/2257 ports). Each leg: a verified restart with the
# given environment, then the decode sweep and the token-hash contract against whatever image the patch set selects.
#   ab-upstream.sh <label>=<VAR=VAL,...> ...      e.g.  ab-upstream.sh base=L2PF=0 l2pf=L2PF=1
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
LOG=docs/speed/ab-upstream.log
say() { printf '%s\n' "$*" | tee -a "$LOG"; }
for leg in "$@"; do
  label=${leg%%=*}; vars=${leg#*=}
  say ""; say "########## $label ($vars)  $(date -u +%Y-%m-%dT%H:%M:%SZ)  patches=$(ls patches/engine/*.patch | wc -l) ##########"
  ( IFS=,; for kv in $vars; do export "$kv"; done
    bash docs/safestart.sh ) >> "$LOG" 2>&1 || { say "  FAILED to start $label"; exit 3; }
  docker ps --filter name=glm53-spark --format '  image={{.Image}}' | tee -a "$LOG"
  LOADS=1,4,8 TRIALS=3 python3 -I docs/dec.py "up-$label" 2>&1 | tee -a "$LOG"
  python3 -I docs/dec.py "up-$label" --contract 2>&1 | tee -a "$LOG"
  say "########## $label done  $(date -u +%H:%M:%SZ) ##########"
done
