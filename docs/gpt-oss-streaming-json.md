# VoxAI vLLM Fork: GPT-OSS Streaming Support

> **Fork maintained by VoxAI**
> This is a fork of [vllm-project/vllm](https://github.com/vllm-project/vllm) with modifications to enable streaming for GPT-OSS models with JSON output.

---

## Why This Fork Exists

The upstream vLLM has two behaviors that break streaming for our GPT-OSS drive-thru use case:

1. **JSON Schema Constraint Buffers Output** - When using `response_format.json_schema`, vLLM routes tokens through xgrammar/guidance backends that buffer until complete valid JSON is formed. Users see nothing until the entire response is ready.

2. **Thinking/Analysis Channel Overhead** - GPT-OSS outputs a `<|channel|>analysis` block before the final response, adding latency we don't need in production.

## What We Changed

| File | Change | Why |
|------|--------|-----|
| `vllm/envs.py` | Added `VLLM_SKIP_THINKING` env var | Skip the analysis channel for faster responses |
| `vllm/envs.py` | Added `VLLM_SKIP_JSON_CONSTRAINT` env var | Bypass JSON schema buffering for streaming |
| `vllm/reasoning/gptoss_reasoning_parser.py` | Added "final" channel to structural tag | Allow final output to stream with `any_text` |
| `vllm/entrypoints/openai/responses/serving.py` | Bypass logic for JSON constraint | When enabled, clears `json` constraint and uses `structural_tag` |

## How to Use

```bash
VLLM_SKIP_THINKING=1 VLLM_SKIP_JSON_CONSTRAINT=1 vllm serve /workspace/gpt-oss-120b --port 8888
```

**Client-side validation**: Since JSON isn't validated server-side, use Pydantic's `experimental_allow_partial="trailing-strings"` to validate streaming JSON on the client. Our `IncrementalExtractor` in assistants handles this.

## Trade-offs

- **Pro**: Tokens stream immediately, much better UX
- **Pro**: No analysis overhead, faster time-to-first-token
- **Con**: Client must handle JSON validation (Pydantic does this well)
- **Con**: Malformed JSON is caught client-side, not server-side (pydantic-ai retries automatically)

---

# GPT-OSS Streaming JSON Output Fix

## Problem Statement

When using GPT-OSS with structured JSON output (via `response_format` with `json_schema`), the response **does not stream**. Instead, the entire JSON is buffered and returned only in the `ResponseCompletedEvent`.

This creates a poor user experience - users see no output until the entire response is generated.

---

## Root Cause Analysis

### How Streaming Works Without JSON Schema

When no JSON schema is provided:

1. Request comes in without `response_format.json_schema`
2. `all_non_structural_tag_constraints_none()` returns `True`
3. `prepare_structured_tag()` is called in `serving.py:470-477`
4. Creates a `structural_tag` with `"content": {"type": "any_text"}`
5. Tokens stream freely via `ResponseOutputTextDeltaEvent`

### How It Breaks With JSON Schema

When JSON schema is provided:

1. Request includes `response_format.type = "json_schema"`
2. In `protocol.py:265-274`, this creates `StructuredOutputsParams(json=schema)`
3. `all_non_structural_tag_constraints_none()` returns `False` (because `json` is set)
4. `prepare_structured_tag()` is NOT called
5. The `json` constraint goes to xgrammar/guidance backend
6. Backend constrains token generation and **buffers until complete valid JSON**
7. Output only appears in `ResponseCompletedEvent` as `McpCall` with `name="<|constrain|>json"`

### The Buffering Location

The buffering happens in the structured output backends:
- `vllm/v1/structured_output/backend_xgrammar.py`
- `vllm/v1/structured_output/backend_guidance.py`

These backends validate each token against the JSON schema grammar and only emit the complete result.

---

## Current Code Flow

### File: `vllm/entrypoints/openai/responses/protocol.py`

Lines 265-274 - Creates the JSON constraint:

```python
# Structured output
structured_outputs = None
if self.text is not None and self.text.format is not None:
    response_format = self.text.format
    if (
        response_format.type == "json_schema"
        and response_format.schema_ is not None
    ):
        structured_outputs = StructuredOutputsParams(
            json=response_format.schema_  # <-- This triggers buffering
        )
```

### File: `vllm/entrypoints/openai/responses/serving.py`

Lines 465-477 - Decides whether to use structural_tag:

```python
if self.reasoning_parser is not None:
    reasoning_parser = self.reasoning_parser(tokenizer)
    if (
        isinstance(
            struct_out := sampling_params.structured_outputs,
            StructuredOutputsParams,
        )
        and struct_out.all_non_structural_tag_constraints_none()  # <-- Returns False when json is set
    ):
        sampling_params.structured_outputs = replace(
            struct_out,
            structural_tag=reasoning_parser.prepare_structured_tag(
                struct_out.structural_tag, self.tool_server
            ),
        )
```

### File: `vllm/reasoning/gptoss_reasoning_parser.py`

Lines 19-33 - Current structural_tag (only defines "analysis" channel):

```python
no_func_reaonsing_tag = {
    "type": "structural_tag",
    "format": {
        "type": "triggered_tags",
        "tags": [
            {
                "begin": "<|channel|>analysis<|message|>",
                "content": {"type": "any_text"},
                "end": "<|end|>",
            }
            # NOTE: No "final" channel defined!
        ],
        "triggers": ["<|channel|>analysis"],
        "stop_after_first": False,
    },
}
```

---

## Solution

### Approach

Instead of using the `json` constraint (which triggers xgrammar/guidance buffering), use a `structural_tag` that includes the "final" channel with `"type": "any_text"`. This allows tokens to stream freely.

JSON validation happens **client-side** using Pydantic's `experimental_allow_partial="trailing-strings"` feature, which can validate incomplete JSON during streaming.

### Why This Works

1. GPT-OSS is trained to output valid JSON when instructed - it doesn't need server-side enforcement
2. The `structural_tag` with `any_text` content allows free token streaming
3. Client-side Pydantic partial validation catches any malformed JSON
4. If validation fails, pydantic-ai retries (existing behavior)

---

## Implementation Plan

### 1. Modify `gptoss_reasoning_parser.py`

Add the "final" channel to `no_func_reaonsing_tag`:

```python
no_func_reaonsing_tag = {
    "type": "structural_tag",
    "format": {
        "type": "triggered_tags",
        "tags": [
            {
                "begin": "<|channel|>analysis<|message|>",
                "content": {"type": "any_text"},
                "end": "<|end|>",
            },
            {  # NEW: Add final channel with any_text
                "begin": "<|channel|>final<|message|>",
                "content": {"type": "any_text"},
                "end": "<|end|>",
            },
        ],
        "triggers": ["<|channel|>analysis", "<|channel|>final"],
        "stop_after_first": False,
    },
}
```

### 2. Modify `serving.py`

Add environment variable `VLLM_SKIP_JSON_CONSTRAINT` to bypass JSON constraint:

```python
import os

SKIP_JSON_CONSTRAINT = os.environ.get("VLLM_SKIP_JSON_CONSTRAINT", "0") == "1"

# In the responses serving code, around line 465:
if self.reasoning_parser is not None:
    reasoning_parser = self.reasoning_parser(tokenizer)

    # Check if we should skip JSON constraint for streaming
    should_use_structural_tag = (
        struct_out.all_non_structural_tag_constraints_none()
        or (SKIP_JSON_CONSTRAINT and struct_out.json is not None)
    )

    if (
        isinstance(
            struct_out := sampling_params.structured_outputs,
            StructuredOutputsParams,
        )
        and should_use_structural_tag
    ):
        # Clear the json constraint if we're bypassing it
        if SKIP_JSON_CONSTRAINT and struct_out.json is not None:
            struct_out = replace(struct_out, json=None)

        sampling_params.structured_outputs = replace(
            struct_out,
            structural_tag=reasoning_parser.prepare_structured_tag(
                struct_out.structural_tag, self.tool_server
            ),
        )
```

### 3. Update `protocol.py` (Optional)

Could also add the env var check here to skip creating the JSON constraint entirely:

```python
# Structured output
structured_outputs = None
if self.text is not None and self.text.format is not None:
    response_format = self.text.format
    if (
        response_format.type == "json_schema"
        and response_format.schema_ is not None
        and not os.environ.get("VLLM_SKIP_JSON_CONSTRAINT")  # NEW
    ):
        structured_outputs = StructuredOutputsParams(
            json=response_format.schema_
        )
```

---

## Usage

Start vLLM with the environment variable:

```bash
VLLM_SKIP_THINKING=1 VLLM_SKIP_JSON_CONSTRAINT=1 vllm serve /workspace/gpt-oss-120b \
  --max-model-len 131072 \
  --port 8888 \
  --tool-call-parser openai \
  --enable-auto-tool-choice
```

---

## Client-Side Changes (assistants-gpt-oss)

With streaming enabled, update the model config to use `IncrementalExtractor`:

```python
# In src/assistants/models.py
"gpt-oss-120b": ModelConfig(
    name="/workspace/gpt-oss-120b",
    model_type="openai-responses",
    settings=OpenAIResponsesModelSettings(
        temperature=0.1,
        openai_reasoning_effort="low",
    ),
    extractor_factory=IncrementalExtractor,  # Changed from EndOnlyExtractor
),
```

The `IncrementalExtractor` uses Pydantic's `experimental_allow_partial="trailing-strings"` to validate incomplete JSON during streaming.

---

## Testing

1. Start vLLM with `VLLM_SKIP_JSON_CONSTRAINT=1`
2. Send a request with `response_format.json_schema`
3. Verify streaming events are received:
   - `ResponseOutputTextDeltaEvent` (should appear during generation)
   - NOT just `ResponseCompletedEvent` with `McpCall`

Test with curl:

```bash
curl -X POST "http://localhost:8888/v1/responses" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/gpt-oss-120b",
    "input": [{"role": "user", "content": "Say hello"}],
    "text": {
      "format": {
        "type": "json_schema",
        "schema": {"type": "object", "properties": {"r": {"type": "string"}}}
      }
    },
    "stream": true
  }'
```

---

## Files Modified

| File | Change |
|------|--------|
| `vllm/reasoning/gptoss_reasoning_parser.py` | Add "final" channel to `no_func_reaonsing_tag` |
| `vllm/entrypoints/openai/responses/serving.py` | Add `VLLM_SKIP_JSON_CONSTRAINT` logic |
| `vllm/entrypoints/openai/responses/protocol.py` | (Optional) Skip creating JSON constraint |

---

## Related

- `docs/gpt-oss-skip-thinking.md` - Similar pattern for skipping reasoning output
- `VLLM_SKIP_THINKING` environment variable
