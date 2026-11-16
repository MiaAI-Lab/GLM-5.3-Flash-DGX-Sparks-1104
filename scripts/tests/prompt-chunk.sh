#!/usr/bin/env bash
# PROMPT_CHUNK (patch 0122, patches/engine/chunk2048/): auto builds 2,048-row prompt passes for DENSE=bf16 on the pair
# and 4,096 everywhere else; 2048 and 4096 choose either; other values are refused. A 4,096-row image's inputs are
# the top-level patches only, so its tag is the one it had before chunk2048/ existed. Each case starts from the
# defaults, given here so that scripts/local.sh and .env cannot change them.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
fails=0
run() { env -i PATH="$PATH" HOME="$HOME" WORKER=user@host DENSE=q8 FP8=0 PROMPT_CHUNK=auto "$@" bash -c '
  source scripts/prepare.sh
  [[ "$PROMPT_CHUNK" =~ ^(auto|2048|4096)$ ]] || validate_config >/dev/null  # only the refusal needs the whole check
  printf "%s %s\n" "$(prompt_chunk)" "$(image_inputs | grep -c "^patches/engine/chunk2048/" || true)"' 2>/dev/null || echo refused; }
check() { # check <name> <want: "rows chunk2048-files" | refused> [VAR=value]...
  local name=$1 want=$2 got
  shift 2
  got=$(run "$@")
  if [[ "$got" == "$want" ]]; then echo "prompt-chunk: $name: pass"; else echo "prompt-chunk: $name: FAIL (got $got, want $want)" >&2; fails=1; fi
}
n=$(find patches/engine/chunk2048 -maxdepth 1 -name '*.patch' | wc -l | tr -d ' ')
check 'q8 (the default) takes 4096' '4096 0'
check 'bf16 on the pair takes 2048' "2048 $n" DENSE=bf16
check 'bf16 at TP=4 takes 4096' '4096 0' DENSE=bf16 GLM_TP4=1
check 'PROMPT_CHUNK=4096 keeps bf16 at 4096' '4096 0' DENSE=bf16 PROMPT_CHUNK=4096
check 'PROMPT_CHUNK=2048 applies to q8' "2048 $n" PROMPT_CHUNK=2048
check 'PROMPT_CHUNK=1024 is refused' refused PROMPT_CHUNK=1024
(( n > 0 )) || { echo 'prompt-chunk: patches/engine/chunk2048/ holds no patch' >&2; fails=1; }
exit $fails
