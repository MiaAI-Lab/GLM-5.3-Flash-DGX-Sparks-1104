#!/usr/bin/env bash
# PROMPT_CACHE_MIN_GIB (issue 16): start.sh reads the prompt cache each rank's startup line funds and stops both ranks
# before LIVE when one funds less. The test runs funded_cache_gib and check_cache_min from start.sh on startup lines
# in the engine's own formats, with Docker and the stop stubbed.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
tmp=$(mktemp -d); trap 'rm -rf -- "$tmp"' EXIT
# The functions alone, from start.sh (sourcing start.sh would run the start).
fns=$(awk '/^(funded_cache_gib|last_archive|check_cache_min)\(\) \{/ { on = 1 } on { print } on && /^\}/ { on = 0 }' start.sh)
[[ "$fns" == *'funded_cache_gib()'* && "$fns" == *'check_cache_min()'* ]] || { echo 'functions not found in start.sh' >&2; exit 1; }
head='[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 83.29 GiB; 8 streams at once, the engine'"'"'s own and 7 more taking 2056 pages of 512 rows (6.26 GiB) as they need them (enough for one at a 1048576-token window), 417 MiB fixed each, of 18.9 GiB left after a 9.5 GiB reserve; prompts in passes of up to 4096 rows'
line() { printf '%s%s; own glue kernels\n' "$head" "$1"; }

# run MIN RANK0_TAIL RANK1_TAIL: prints check_cache_min's exit status, then what it logged, warned, failed and stopped.
run() {
  local logs="$tmp/logs.$RANDOM"
  local d="$logs/run-1"; mkdir -p "$d"
  [[ "$2" == - ]] || line "$2" > "$d/rank0.log"
  [[ "$3" == - ]] || line "$3" > "$d/rank1.log"
  # shellcheck disable=SC2034,SC2329 # the variables and stubs are read by the functions eval'd below
  (
    set +e
    PROMPT_CACHE_MIN_GIB=$1 RUN_LOG_DIR=$d LOG_DIR=$logs
    collect_logs() { :; }
    # as stop.sh: archive the run directory, then remove it
    stop_both() { echo stopped; tar -czf "$LOG_DIR/glm53-spark-1-1.tar.gz" -C "$RUN_LOG_DIR" . && rm -rf -- "$RUN_LOG_DIR"; }
    log() { echo "log: $*"; }
    warn() { echo "warn: $1"; }
    fail() { echo "fail: $1 logs=${2##*Logs: }"; exit 1; }
    at_least() { python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)' "$1" "$2"; }
    eval "$fns"
    check_cache_min
    echo "status $?"
  ) 2>&1 | sed "s#$tmp/logs\.[0-9]*#LOGS#g" | tr '\n' '|'
}

check() { # check NAME GOT WANT
  if [[ "$2" == "$3" ]]; then echo "cache-min: $1: pass"; else echo "cache-min: $1: got [$2], want [$3]" >&2; exit 1; fi
}

eval "$fns"
f="$tmp/one.log"
line ', a 9.7 GiB prompt cache' > "$f"; check 'device cache read' "$(funded_cache_gib "$f")" 9.7
line ', no prompt cache' > "$f"; check 'no cache is 0' "$(funded_cache_gib "$f")" 0
line ', a 3.0 GiB prompt cache in host memory with 64 MiB of device staging' > "$f"; check 'host cache is 0 on the device' "$(funded_cache_gib "$f")" 0
: > "$f"; check 'no startup line is empty' "$(funded_cache_gib "$f")" ''

check 'off when empty' "$(run '' ', a 1.0 GiB prompt cache' ', a 1.0 GiB prompt cache')" 'status 0|'
check 'both ranks fund enough' "$(run 9 ', a 9.7 GiB prompt cache' ', a 12.9 GiB prompt cache')" \
  'log: Prompt cache: rank 0 funds 9.7 GiB (PROMPT_CACHE_MIN_GIB=9).|log: Prompt cache: rank 1 funds 12.9 GiB (PROMPT_CACHE_MIN_GIB=9).|status 0|'
check 'rank 1 short stops both' "$(run 9 ', a 9.7 GiB prompt cache' ', a 8.7 GiB prompt cache')" \
  'log: Prompt cache: rank 0 funds 9.7 GiB (PROMPT_CACHE_MIN_GIB=9).|stopped|fail: E_CACHE logs=LOGS/glm53-spark-1-1.tar.gz.|'
check 'no cache stops both' "$(run 1 ', no prompt cache' ', a 9.7 GiB prompt cache')" 'stopped|fail: E_CACHE logs=LOGS/glm53-spark-1-1.tar.gz.|'
check 'rank 0 without a startup line stops both' "$(run 9 - ', a 9.7 GiB prompt cache')" 'stopped|fail: E_CACHE logs=LOGS/glm53-spark-1-1.tar.gz.|'
check 'rank 1 without a startup line is rank 0 only, no warning' "$(run 9 ', a 9.7 GiB prompt cache' -)" \
  'log: Prompt cache: rank 0 funds 9.7 GiB (PROMPT_CACHE_MIN_GIB=9).|status 0|'
