# Issue: tool-call suppression and malformed envelopes (Hermes Agent report, 2026-11-06)

| | |
|---|---|
| **Report** | "Tool-call suppression and malformed tool-call envelopes — comprehensive report", Hermes Agent session, 2026-11-06 |
| **Status** | Engine patch 0112 written and host-tested (2026-11-09); not yet built into the pair's image or soaked on live traffic |
| **Related** | CHANGELOG 2026-11-07 (top_k 20, `EFFORT_LINE=thinking`), patches 0103, 0107 |

## Summary

The report is right that the guard's offered-tools path made a visible failure silent, and that the logs could not
diagnose it. Patch 0112 fixes both. Two of its claims did not hold up against the code (T2/E6 and part of T3, below).
The malformed-envelope rate it measured (5.7%, Nov 5-6) predates the two changes of 2026-11-07, so it has to be
measured again before anything else is changed.

## What changed (patch 0112)

- **One log line per rejected envelope (T1, T4, T6, E8):**
  `tool call rejected from chatcmpl-…: REASON ['NAME'] (N tools offered) thinking=… effort=…; FATE`.
  Reasons: `unknown_tool_name` (parses as a call; the request did not offer that name), `malformed` (no spelling the
  parser reads: no name, a name over two lines, broken JSON), `unclosed` (the end token came before `</tool_call>`;
  before 0112 this case was not logged at all). `FATE` is `kept as text`, `hidden (TF_TOOL_CALL_GUARD)`, or
  `kept as text (hiding it would leave the reply empty)`.
- **`TF_TOOL_CALL_DEBUG=1`** adds the envelope text (escaped to one line, cut at about 1 KB) after each line (T4).
- **The guard with tools offered (T3, T7)** hides only well-formed calls to unoffered tools. Malformed calls stay as
  text, as with the guard off. If hiding would leave a reply with no text and no call, nothing is hidden, so the
  empty-reply / blind-retry loop of E2 cannot occur. The no-tools path (the guard's documented purpose) is unchanged.
- **Host tests (T10):** rejection reasons for each shape, a mixed good/unoffered/malformed batch under the guard,
  the Hermes bridge call, and HTTP replies with the guard on, whole and streamed. They run in
  `zig build test-tool-parse-host` and `test-tool-visibility-http-host`.

## Claims that did not hold

- **T2 / E6, "the parser forwards nameless calls as structured `tool_calls`":** every path that builds a
  `tool_calls` entry checks the name against the offered tools (`callName` in `tool_parse.zig`), and a nameless JSON
  call is rejected (now tested). The client error `Tool tool_call returned error: tool_call calls[0] requires a 'name'`
  comes from Hermes' own bridge tool named **`tool_call`** (`hermes-agent/tools/tool_search_validation.py:172`),
  which invokes deferred tools via `calls: [{name, arguments}]`. The server correctly forwarded a call to that offered
  tool. The model left `name` out of an entry inside its arguments, and Hermes refused it. That is why no server
  rejection line accompanies it.
- **T3, "prose is discarded when envelopes are hidden":** `hideCalls` kept the surrounding prose (tested
  since 0107). Reasoning is split into `reasoning_content` before calls are parsed, so the large token counts of the
  14 replies were most likely reasoning, which the guard did not touch (not verified against those replies). The real
  defect was the emptied `content`, which 0112 prevents.
- **Lenient GLM parsing:** outside `parallel_tool_calls: false`, `NAME<arg_key><arg_value>V</arg_value>` (key name
  missing, the 2026-11-07 symptom when it has its closer) is accepted as a call to NAME with empty arguments, not
  rejected. That is upstream behaviour, kept unchanged. It shows up as a tool error on the client, with no rejection
  line on the server.

## Root cause of the malformed rate: what is known

- On 2026-11-07, the template stopped writing `Reasoning Effort: Max` into thinking-off prompts
  (`EFFORT_LINE=thinking`, now the default). In a copy test, that cut malformed markup from 30% to 0.2%
  (3 of 1,419 lines). The pair also serves `--top-k 20` since then. Both came after the report's window (Nov 5-6).
- Thinking-on turns (23 of the report's 25 events) still get the effort line, as the checkpoint template writes it.
  The report's A/B suggestions (T5/T8: `clear_thinking=false` for tool histories, `reasoning_effort: high` for agent
  loops) remain open and should be measured with the 0112 log lines rather than inferred from counts.
- T9 (retired recipe pin) is not actionable here; the weights revision before `76c0b517` is not recoverable from this
  repo.

## To do on the pair

1. `./start.sh restart` (the patch set changed, so the image is rebuilt). Optionally add
   `export TF_TOOL_CALL_DEBUG=1` to `scripts/local.sh` for the soak.
2. A day of Hermes traffic, then:
   `docker logs glm53-spark 2>&1 | grep -o "tool call rejected from [^:]*: [a-z_]*" | awk '{print $NF}' | sort | uniq -c`
   gives rejections by reason; compare with `docker logs glm53-spark 2>&1 | grep -c "\] done chatcmpl"` completions,
   and split by `effort=` with `grep -o "effort=[a-z]*; .*"`.
3. Only if `malformed` stays well above zero on `effort=max` turns, run the T8 A/B.
