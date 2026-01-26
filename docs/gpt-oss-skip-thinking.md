# GPT-OSS-120B Skip-Thinking Hack

This document explains how to completely disable the "thinking" (reasoning) phase in GPT-OSS-120B when using vLLM's Responses API or Chat Completions API.

## The Problem

GPT-OSS-120B uses the **Harmony Protocol** - a special token format where the model generates output in "channels":

```
<|start|>assistant<|channel|>analysis<|message|>thinking here...<|end|><|channel|>final<|message|>actual response<|end|><|return|>
```

- **analysis channel** = the model's internal thinking (what we want to skip)
- **final channel** = the actual response to the user

Normally the model generates 50-100+ tokens of "thinking" before responding. This adds latency and token costs.

## The Solution: 8 Patches

The hack requires patching both **non-streaming** and **streaming** code paths.

### Patch 1: Force the Model to Skip Thinking

**File:** `vllm/entrypoints/openai/parser/harmony_utils.py`

**Function:** `render_for_completion()`

**What it does:** This function creates the prompt tokens sent to the model. We inject special tokens at the END of the prompt:

```python
def render_for_completion(messages: list[Message]) -> list[int]:
    conversation = Conversation.from_messages(messages)
    token_ids = get_encoding().render_conversation_for_completion(
        conversation, Role.ASSISTANT
    )
    # HACK: Inject skip-thinking tokens
    import os
    if os.environ.get("VLLM_SKIP_THINKING", "0") == "1":
        skip_tokens = [200005, 17196, 200008]  # <|channel|>final<|message|>
        token_ids = token_ids + skip_tokens
    return token_ids
```

**Why it works:** When the model sees the prompt ending with `<|channel|>final<|message|>`, it thinks "I'm already in the final channel" and generates the response directly without thinking first.

**Token meanings:**
- `200005` = `<|channel|>`
- `17196` = `final`
- `200008` = `<|message|>`

---

### Patch 2: Fix Parser for Chat Completions API

**File:** `vllm/entrypoints/openai/parser/harmony_utils.py`

**Function:** `parse_output_into_messages()`

**The problem:** After Patch 1, the model generates:
```
Hello! How can I assist you today?<|return|>
```

But the parser expects:
```
<|channel|>final<|message|>Hello! How can I assist you today?<|end|><|return|>
```

The parser doesn't see the channel markers because they're in the PROMPT, not the OUTPUT.

**The fix:** Pre-initialize the parser with the skip-thinking tokens:

```python
def parse_output_into_messages(token_ids: Iterable[int]) -> StreamableParser:
    parser = get_streamable_parser_for_assistant()
    token_ids_list = list(token_ids)
    import os
    import logging
    logger = logging.getLogger(__name__)

    # HACK: When skip-thinking is enabled, pre-process the parser with
    # channel tokens so it starts in the right state to receive content
    if os.environ.get("VLLM_SKIP_THINKING", "0") == "1":
        skip_tokens = [200005, 17196, 200008]  # <|channel|>final<|message|>
        for t in skip_tokens:
            parser.process(t)

    for i, token_id in enumerate(token_ids_list):
        try:
            parser.process(token_id)
        except Exception as e:
            if os.environ.get("VLLM_SKIP_THINKING", "0") == "1":
                break  # Stop parsing, use what we have
            raise
    return parser
```

---

### Patch 3: Fix Parser for Responses API

**File:** `vllm/entrypoints/context.py`

**Function:** `HarmonyContext.append_output()`

The Responses API uses a different code path. Apply the same pre-processing:

```python
def append_output(self, output: RequestOutput) -> None:
    output_token_ids = output.outputs[0].token_ids
    self.parser = get_streamable_parser_for_assistant()

    # HACK: When skip-thinking is enabled, pre-process the parser with
    # channel tokens so it starts in the right state to receive content
    import os
    if os.environ.get("VLLM_SKIP_THINKING", "0") == "1":
        skip_tokens = [200005, 17196, 200008]
        for t in skip_tokens:
            self.parser.process(t)

    for token_id in output_token_ids:
        try:
            self.parser.process(token_id)
            self._update_num_reasoning_tokens()
        except Exception as e:
            import os
            if os.environ.get("VLLM_SKIP_THINKING", "0") == "1":
                break
            raise
    # ... rest of method unchanged
```

---

### Patch 4: Handle Parser Errors Gracefully

**The problem:** After generating the response, the model sometimes tries to "backfill" thinking:
```
Hello!<|end|><|channel|>analysis<|message|>Let me think...
```

The parser crashes because after `<|end|>` it expects `<|start|>` (new turn), not `<|channel|>`.

**The fix:** Already included in Patches 2 and 3 - catch the error and use whatever was already parsed:

```python
try:
    parser.process(token_id)
except Exception as e:
    if os.environ.get("VLLM_SKIP_THINKING", "0") == "1":
        break  # Stop parsing, use what we have ("Hello!")
    raise
```

---

### Patch 5: Add Stop Tokens (Optional Safety)

**File:** `vllm/entrypoints/openai/parser/harmony_utils.py`

**Function:** `get_stop_tokens_for_assistant_actions()`

Add `<|channel|>` and `<|end|>` as stop tokens to potentially stop generation earlier:

```python
def get_stop_tokens_for_assistant_actions() -> list[int]:
    import os
    stop_tokens = get_encoding().stop_tokens_for_assistant_actions()
    # HACK: Add <|channel|> and <|end|> as stop tokens when skip-thinking
    if os.environ.get("VLLM_SKIP_THINKING", "0") == "1":
        stop_tokens = list(stop_tokens) + [200005, 200007]  # <|channel|>, <|end|>
    return stop_tokens
```

---

### Patch 6: Fix Streaming for Responses API

**File:** `vllm/entrypoints/context.py`

**Class:** `StreamingHarmonyContext.__init__()`

Streaming uses a different context class. Add pre-processing in the constructor:

```python
class StreamingHarmonyContext(HarmonyContext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_output = None

        self.parser = get_streamable_parser_for_assistant()

        # HACK: When skip-thinking is enabled, pre-process the parser
        import os
        if os.environ.get("VLLM_SKIP_THINKING", "0") == "1":
            skip_tokens = [200005, 17196, 200008]  # <|channel|>final<|message|>
            for t in skip_tokens:
                self.parser.process(t)

        self.encoding = get_encoding()
        # ... rest unchanged
```

Also add error handling in `StreamingHarmonyContext.append_output()`:

```python
for tok in output.outputs[0].token_ids:
    try:
        self.parser.process(tok)
        last_delta_text += self.parser.last_content_delta or ""
    except Exception as e:
        import os
        if os.environ.get("VLLM_SKIP_THINKING", "0") == "1":
            break
        raise
```

---

### Patch 7: Fix Streaming for Chat Completions API

**File:** `vllm/entrypoints/openai/serving_chat.py`

Chat Completions streaming creates its own parsers. Find the `harmony_parsers` initialization and add pre-processing:

```python
if self.use_harmony:
    harmony_parsers = []
    for _ in range(num_choices):
        parser = get_streamable_parser_for_assistant()
        # HACK: When skip-thinking is enabled, pre-process parser
        import os
        if os.environ.get("VLLM_SKIP_THINKING", "0") == "1":
            skip_tokens = [200005, 17196, 200008]
            for t in skip_tokens:
                parser.process(t)
        harmony_parsers.append(parser)
    harmony_tools_streamed = [False] * num_choices
```

---

### Patch 8: Error Handling in Chat Completions Streaming

**File:** `vllm/entrypoints/openai/serving_chat.py`

Add error handling in the token processing loop:

```python
delta_text = ""
for token_id in output.token_ids:
    try:
        harmony_parser.process(token_id)
        delta_text += harmony_parser.last_content_delta or ""
    except Exception as e:
        import os
        if os.environ.get("VLLM_SKIP_THINKING", "0") == "1":
            break
        raise
cur_channel = harmony_parser.current_channel
```

---

## Visual Summary

**Normal flow (with thinking):**
```
Prompt: "Say hello"
         |
         v
Model generates: <|channel|>analysis<|message|>The user wants a greeting...<|end|>
                 <|channel|>final<|message|>Hello!<|end|><|return|>
         |
         v
Output: "Hello!" (but 50+ thinking tokens wasted)
```

**Skip-thinking flow:**
```
Prompt: "Say hello" + <|channel|>final<|message|>  <-- INJECTED
         |
         v
Model generates: Hello!<|return|>  <-- Goes straight to response
         |
         v
Parser pre-initialized with: <|channel|>final<|message|>  <-- KNOWS IT'S FINAL
         |
         v
Output: "Hello!" (0 thinking tokens!)
```

---

## Files Modified Summary

| File | Functions/Classes Modified |
|------|---------------------------|
| `vllm/entrypoints/openai/parser/harmony_utils.py` | `render_for_completion()`, `parse_output_into_messages()`, `get_stop_tokens_for_assistant_actions()` |
| `vllm/entrypoints/context.py` | `HarmonyContext.append_output()`, `StreamingHarmonyContext.__init__()`, `StreamingHarmonyContext.append_output()` |
| `vllm/entrypoints/openai/serving_chat.py` | `harmony_parsers` initialization, token processing loop |

---

## Usage

```bash
# Start vLLM with the environment variable to enable skip-thinking
VLLM_SKIP_THINKING=1 vllm serve /path/to/gpt-oss-120b --host 0.0.0.0 --port 8000

# Without the env var, normal behavior (with thinking)
vllm serve /path/to/gpt-oss-120b --host 0.0.0.0 --port 8000
```

---

## Test Results

| API | Mode | Status | Content | Reasoning Tokens |
|-----|------|--------|---------|------------------|
| Chat Completions | Non-streaming | Works | "Hello! How can I assist you today?" | 0 |
| Chat Completions | Streaming | Works | Token-by-token deltas | 0 |
| Responses | Non-streaming | Works | "Hello! How can I assist you today?" | 0 |
| Responses | Streaming | Works | Token-by-token deltas | 0 |

---

## Harmony Protocol Token Reference

| Token ID | Symbol | Meaning |
|----------|--------|---------|
| 200005 | `<|channel|>` | Start channel name |
| 200006 | `<|start|>` | Start new turn |
| 200007 | `<|end|>` | End current message |
| 200008 | `<|message|>` | Start message content |
| 200002 | `<|return|>` | End turn, return to user |
| 200012 | `<|call|>` | Tool call |
| 17196 | `final` | Final channel name (text) |
| 35644 | `analysis` | Analysis channel name (text) |

---

## Limitations

- This is a hack, not an official feature
- May break with future vLLM updates
- The model may occasionally produce lower quality responses without thinking
- Tool calls have not been extensively tested with this hack
