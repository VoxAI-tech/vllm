# vLLM Patches for GPT-OSS 120B

**Last updated:** 2026-01-30
**Server:** RunPod B200

---

## Files

| File | Description |
|------|-------------|
| `serving_responses.py` | Main responses API - handles JSON streaming for all channel/recipient variants |
| `serving_chat_stream_harmony.py` | Stream indexing for Harmony protocol |
| `harmony_utils.py` | Output parsing for Harmony protocol |
| `harmony_utils_working_backup.py` | Original backup before patches |
| `harmony_client.py` | Test client for Harmony protocol |

---

## vLLM Startup Command

```bash
cd /workspace && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VLLM_SKIP_THINKING=1 \
vllm serve /workspace/gpt-oss-120b \
  --port 8888 \
  --reasoning-parser openai_gptoss \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.92
```

---

## File Locations on Server

```
/workspace/venv/lib/python3.12/site-packages/vllm/entrypoints/openai/
├── serving_responses.py
├── serving_chat_stream_harmony.py
└── parser/
    └── harmony_utils.py
```

---

## Patch Purpose

Enable JSON streaming for GPT-OSS model output across all Harmony protocol variants.

### Problem

GPT-OSS outputs `final_result` JSON inconsistently across different channel/recipient combinations:

1. `recipient=<|constrain|>json` (standard constraint)
2. `channel=final_result`, `recipient=None`
3. `channel=final`, `recipient=None` (raw JSON content)

Stock vLLM only recognized `functions.X` recipients, causing:
- JSON buffered instead of streamed
- Duplicate output (both function call AND text events)
- Concatenated responses with embedded JSON

### Solution

Patch all three files to recognize constraint recipients and JSON content as function calls:

**harmony_utils.py:**
- Fixed indentation bug in `parse_remaining_state()` (was causing syntax error)
- Handle constraint recipients regardless of channel

**serving_responses.py:**
- `_emit_content_delta_events()`: Route `channel=final_result` to function call events
- `_emit_content_delta_events()`: Route `channel=final` + constraint recipients to function call events
- `_emit_previous_item_done_events()`: Handle `channel=final_result` as function call
- `_emit_previous_item_done_events()`: Detect JSON content (`{"r":...}`) on final channel and emit as function call

**serving_chat_stream_harmony.py:**
- Extended `is_constraint_recipient` check to include `<|constrain|>json` and `<|channel|>commentary`

---

## Deployment

Copy patches to server:
```bash
SSH_CMD="ssh -p <PORT> -i ~/.ssh/id_ed25519 root@66.92.198.130"
VLLM_PATH="/workspace/venv/lib/python3.12/site-packages/vllm/entrypoints/openai"

scp -P <PORT> -i ~/.ssh/id_ed25519 harmony_utils.py root@66.92.198.130:$VLLM_PATH/parser/
scp -P <PORT> -i ~/.ssh/id_ed25519 serving_responses.py root@66.92.198.130:$VLLM_PATH/
scp -P <PORT> -i ~/.ssh/id_ed25519 serving_chat_stream_harmony.py root@66.92.198.130:$VLLM_PATH/
```

Restart vLLM after copying.

---

## Environment Variables

```bash
DRIVETHRU_MODEL=gpt-oss-120b
OPENAI_BASE_URL=https://<pod-id>-8888.proxy.runpod.net/v1
RAG_URL=http://127.0.0.1:9000
TWIN_MODEL=gemini-2.5-flash
```
