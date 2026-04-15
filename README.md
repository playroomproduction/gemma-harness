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

## Stability Tuning Test (Before vs After)

Test date: `2026-04-15`  
Hardware: `Mac mini M4 16GB`  
Workload: long prompt smoke test through harness `/chat`

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

### Evidence

- Failure signal in MLX log (before):
  - `Command buffer execution failed: Insufficient Memory ... OutOfMemory`
- Success signal (after):
  - long-prompt smoke returned:
  - `ok True rounds 1 tools 0 latency 41.98`

## Notes

- This repo tracks harness-level logic and defaults.
- Machine-specific LaunchAgent flags (local service profile) live outside this repo and are managed on-host.
