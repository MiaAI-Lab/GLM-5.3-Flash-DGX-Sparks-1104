#!/usr/bin/env bash
# Each launcher must send one explicit value to every rank, even when the recipe default is unexported.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

check() (
  local launcher=$1 value=$2 want=$2 arg
  local -a matches=()
  unset TF_GLM_PROMPT_WIDE
  if [[ "$value" == default ]]; then want=0; else export TF_GLM_PROMPT_WIDE="$value"; fi
  source scripts/config.sh
  # Load the launcher's environment builder without starting a server.
  # shellcheck disable=SC1090
  source <(sed -n -e '/^bits()/p' -e "/^MANAGED_ENV=/,/'$/p" -e '/^build_env_args()/,/^}/p' "$launcher")
  LINK_SELF=127.0.0.1 MASTER_ADDR=127.0.0.1 ROCE_RING=off PREFILL_REDUCE=0 build_env_args
  for arg in "${ENV_ARGS[@]}"; do
    case "$arg" in TF_GLM_PROMPT_WIDE|TF_GLM_PROMPT_WIDE=*) matches+=("$arg");; esac
  done
  [[ "${#matches[@]}" == 1 && "${matches[0]}" == "TF_GLM_PROMPT_WIDE=$want" ]]
)

for launcher in start.sh start-tp4-switchless.sh; do
  for value in default 0 1; do check "$launcher" "$value"; done
  echo "prompt-wide-env ($launcher): pass"
done
