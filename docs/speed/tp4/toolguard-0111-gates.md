# 0111 TF_TOOL_CALL_GUARD gates (registered 2026-11-05, before the build was tested)
- BUILD: image builds with 0111.
- OFF (default, guard unset): cases B/C/D still return the markup as content (behaviour unchanged); case A 2 tool_calls;
  acc.py hashes equal the image's known ones (FP8 b363136de28e / d240a39d8c68).
- ON (TF_TOOL_CALL_GUARD=1): B/C/D, streamed and whole: no "<tool_call>" / "<arg_key>" in content, finish not tool_calls;
  case A (tools offered) still 2 tool_calls, streamed and whole; acc.py hashes unchanged (no tool history -> no note);
  a server log line "tool-call guard: hid ..." when markup was hidden.
