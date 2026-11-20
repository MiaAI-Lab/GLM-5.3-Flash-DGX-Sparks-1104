#!/usr/bin/env bash
# Refuse unsupported committed-snapshot settings before restart can stop a serving pair.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
fails=0
check() {
  local name=$1 want=$2 err rc got
  shift 2
  err=$(mktemp)
  env -i PATH="$PATH" HOME="$HOME" WORKER=user@host TP=2 KV_SHARD=0 LEARN_GIB=0 "$@" bash -c '
    source scripts/prepare.sh
    validate_config >/dev/null' 2>"$err" && rc=0 || rc=$?
  if (( rc == 0 )); then got=accepted
  elif grep -q 'E_CONFIG.*TF_GLM_SNAPSHOT_FORMAT' "$err"; then got=refused
  else got="error $rc: $(head -1 "$err")"; fi
  rm -f -- "$err"
  if [[ "$got" == "$want" ]]; then echo "compact-snapshot: $name: pass"; else echo "compact-snapshot: $name: FAIL (got $got, want $want)" >&2; fails=1; fi
}
check 'unset retains legacy' accepted
check 'explicit legacy' accepted TF_GLM_SNAPSHOT_FORMAT=legacy
check 'committed on the pair' accepted TF_GLM_SNAPSHOT_FORMAT=committed
check 'decimal zero learn budget' accepted TF_GLM_SNAPSHOT_FORMAT=committed LEARN_GIB=0.0
check 'disk learning refused' refused TF_GLM_SNAPSHOT_FORMAT=committed LEARN_GIB=1
check 'fractional disk learning refused' refused TF_GLM_SNAPSHOT_FORMAT=committed LEARN_GIB=0.1
check 'KV sharding refused' refused TF_GLM_SNAPSHOT_FORMAT=committed KV_SHARD=1
check 'four ranks refused' refused TF_GLM_SNAPSHOT_FORMAT=committed TP=4
check 'one rank refused' refused TF_GLM_SNAPSHOT_FORMAT=committed TP=1
check 'typo refused' refused TF_GLM_SNAPSHOT_FORMAT=comitted
check 'legacy disk learning retained' accepted TF_GLM_SNAPSHOT_FORMAT=legacy LEARN_GIB=1
exit "$fails"
