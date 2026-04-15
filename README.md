# gemma-harness

Claude-style agentic harness for local Gemma 4 (MLX / `vmlx-serve`).

## What This Repo Does

- Wraps local Gemma with an agent loop (`perceive -> think -> act -> observe -> reflect`)
- Adds tool-calling, verification, and context management
- Exposes API endpoints via FastAPI (`/chat`, `/invoke`, `/health`, `/tools`)

## Runtime (Mac mini M4 16GB)

- Model: `mlx-community/gemma-4-e4b-it-4bit`
- MLX server: `127.0.0.1:8091`
- Harness server: `127.0.0.1:8093`

## Runtime Stability Tuning (Before vs After)

Test date: `2026-04-15`  
Hardware: `Mac mini M4 16GB`  
Workload: long prompt smoke test through harness `/chat`

This comparison is about **stability on 16GB hardware**, not absolute model quality.
The "after" profile is more conservative on purpose: it trades some cache precision
and concurrency for a much lower chance of MLX / Metal crashing under long prompts.

| Item | Before (aggressive) | After (stable profile) |
|---|---|---|
| Harness context budget | 22000 | 18000 |
| Harness max output tokens | 1536 | 1536 |
| KV cache quantization | q8 | q4 |
| Paged cache blocks | 1600 | 900 |
| max_num_seqs | 256 | 1 |
| prefill/completion batch | 8 / 32 | 1 / 1 |
| Result on long prompt | Failed (`HTTP 500`, MLX disconnected) | Passed |
| Low-level error | `METAL ... OutOfMemory` | No OOM in smoke test |

### How to Read This Table

- `18000` context budget is **smaller than 22000**, so it is not "stronger" in isolation.
- `q4` KV cache is **lighter than q8**, so it is also not "higher quality" in isolation.
- The point of the change is: **the previous settings were too aggressive for M4 16GB and crashed**.
- In practice, a slightly smaller context that actually completes is more useful than a larger one that dies mid-run.

### Evidence

- Failure signal in MLX log (before):
  - `Command buffer execution failed: Insufficient Memory ... OutOfMemory`
- Success signal (after):
  - long-prompt smoke returned:
  - `ok True rounds 1 tools 0 latency 41.98`

## Harness Capability Gains

The table above only covers runtime stability. The harness itself adds capability that
plain local Gemma does not reliably provide on its own.

| Capability | Plain local model | This harness |
|---|---|---|
| Tool-calling | Fragile / model-dependent | Native tool-calling + text fallback parsing |
| Tool name handling | Exact-match only | Tool-name normalization (`google_search -> web_search`, etc.) |
| Tool hesitation recovery | Usually stalls | Auto-detects hesitation and nudges or injects tool results |
| File hallucination guard | No guard | Detects fake file reads and forces real `read_file` |
| Context handling | Raw history only | Budgeted context manager with trimming and memory recall |
| Execution memory | Hidden in raw turns | Injected execution brief with task, progress, and recent tool results |
| Final answer quality control | None | Verification pass before returning final answer |
| Tool prompt size | Full tool list every time | Task-based tool selection to reduce prompt bloat |

### Tool-Calling Notes

- `vmlx-serve` is configured with native Gemma 4 tool-calling support.
- The harness also supports a fallback parser for text-style tool calls such as:
- `<execute_tool>web_search(...)</execute_tool>`
- This makes Gemma 4 much more usable as an agent than plain single-pass chat.

## Notes

- This repo tracks harness-level logic and defaults.
- Machine-specific LaunchAgent flags (local service profile) live outside this repo and are managed on-host.
