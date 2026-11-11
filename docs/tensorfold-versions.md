# TensorFold versions — what runs where (verified 2026-11-10)

Three pins exist in this ecosystem. They are independent on purpose; do not "sync them blindly".

| Surface | Commit | Date | What it is |
|---|---|---|---|
| **Upstream `ashhart/TensorFold` latest release** | `6e9af93` (tag `v1.2.6`) | 2026-11-10 | The engine upstream. Newer than any pin below. |
| **This recipe's GLM engine pin** | `f350cd3e` (2026-10-30) | scripts/config.sh `TF_REF` | The commit this recipe's 112-patch stack applies to, 112/112 verified sequentially (`git apply` against a pristine tree at the pin). |
| **Imagination Studio's H3 engine** | `1f7ecc75` (2026-10-14) | `runtime/tf-h3/commit` on the deploy | The `libtf_h3.so` the Studio's H3 renders run through (independent build from the GLM server). |

## Why the recipe pin is not "latest"

The patch stack (`patches/engine/*.patch`) is a sequential series on top of `f350cd3e`. Applied against upstream
`v1.2.6` (`6e9af93`) the first patch already fails to apply: upstream's EXL3/CUDA files moved after `75e80ede`
(2026-11-09 20:58 UTC, the newest commit where the full stack still applies cleanly — verified by sequential
`git apply` across every commit since the pin).

**Porting the stack to a newer pin is a real job** (rebase 112 patches, re-verify the DFlash2 pairing, rebuild
both ranks) and is NOT required for the running deployment: the GLM pair on spark-faea/f95e is built from the
pinned image and must not be restarted for an engine refresh. Upgrade only with a maintenance window and a fresh
`docker build` + `vm_list`-style staged swap, per the recipe README.

## Upgrade procedure (when a window is approved)

1. `git -C <probe> checkout <new-pin>` then `for p in patches/engine/*.patch; git apply --check $p` sequentially —
   stop at the first failure and rebase from there.
2. Bump `TF_REF` in `scripts/config.sh`; the image tag hash changes automatically (README §"All hashed into the
   image tag").
3. Rebuild, verify `--health`, then swap one rank at a time (rank1 first, then rank0) with the same
   identity-capture discipline as any GLM lifecycle action (see SAFETY-STATUS.md's watchdog requirements).

## Studio's H3 engine (the render-fit blocker context)

The Studio renders through `libtf_h3.so` at `1f7ecc75` — a different binary than the GLM server. Its memory
guard (`spark/memory_guard.py`) measures `MemAvailable`-style free unified memory **after** the GLM pair's
resident pages, which is why every H3 compile on spark-faea reports `fits:false` while GLM serves
(~57.6 GB int8 base vs ~6 GB free). See `kitty-heist-20261109/handover/tf-fix-report.md` for the verified math
and options; engine-version bumps alone do not change that guard's arithmetic.

## Verification record (2026-11-10)

- `git apply` sequential check at `f350cd3e`: 112/112 clean (this recipe's pin is self-consistent).
- Sequential check at `6e9af93` (v1.2.6): fails at `0001-…mcg-experts…patch` — upstream moved EXL3 CUDA files.
- Newest commit with a clean full-stack apply: `75e80ede` (2026-11-09).
- Upstream latest release: `v1.2.6` / `6e9af93` (2026-11-10T12:51Z); main head at check time `14dea6c6`.
