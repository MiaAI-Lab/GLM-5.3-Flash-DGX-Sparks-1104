#!/usr/bin/env bash
# Serve GLM-5.3-Flash on four DGX Sparks on a switch (TP=4, experimental, not yet run on a switch with this engine):
# ./start-tp4-switchless.sh with FABRIC_EXPECT=mesh. The same image, patches, settings, checks and weights (NFS from
# this Spark, or each worker's own copy) as the ring; what differs is the fabric. Every pair of Sparks must share a
# RoCE subnet (their ConnectX-7 ports on one switch), WORKER, WORKER2 and WORKER3 may be in any order, NCCL picks its
# own algorithms and devices (no ring graphs, no NCCL_ALGO=Ring, no subnet-aware routing), and NCCL carries every
# exchange (ROCE_RING, the ring's RoCE relay, is refused here). Four Sparks cabled to each other without a switch (a
# ring): ./start-tp4-switchless.sh; this script stops there and says so. README: "4 Sparks".
#
# Usage: ./start-tp4.sh [restart]   (./stop-tp4.sh stops it; DRY_RUN=1 ./start-tp4.sh prints the links it finds and every
# rank's command, and changes nothing)
set -euo pipefail
case "${1:-}" in
  help|-h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
esac
export FABRIC_EXPECT=mesh
exec "$(dirname -- "$(readlink -f -- "$0")")/start-tp4-switchless.sh" "$@"
