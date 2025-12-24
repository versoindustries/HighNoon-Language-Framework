# Inference API Reference

> HighNoon Language Framework - Generation and Inference

This document covers text generation, streaming, and advanced inference features.

---

## Basic Generation

Generate text using the `generate()` method on any HighNoon model.

```python
import highnoon as hn

model = hn.create_model("highnoon-3b")

response = model.generate(
    "Explain quantum computing in simple terms",
    max_length=256,
    temperature=0.7,
)
print(response)
```

### Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` or `Tensor` | *required* | Input text or token IDs |
| `max_length` | `int` | `256` | Maximum tokens to generate |
| `temperature` | `float` | `1.0` | Sampling temperature (higher = more random) |
| `top_k` | `int` | `50` | Limit to top-k tokens per step |
| `top_p` | `float` | `1.0` | Nucleus sampling threshold |
| `repetition_penalty` | `float` | `1.0` | Penalty for repeated tokens |
| `stop_sequences` | `list[str]` | `None` | Sequences that stop generation |
| `tokenizer` | `Tokenizer` | `None` | Tokenizer for encoding/decoding |

---

## Streaming Generation

Generate text with streaming output for real-time display.

```python
from highnoon.inference import StreamingGenerator

model = hn.create_model("highnoon-3b")
streamer = StreamingGenerator(model)

for chunk in streamer.generate_stream(
    "Write a short story about",
    max_length=500,
    temperature=0.8,
):
    print(chunk, end="", flush=True)
```

### StreamingGenerator

```python
from highnoon.inference import StreamingGenerator

streamer = StreamingGenerator(
    model,
    chunk_size=8,
    tokenizer=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `HSMN` | *required* | HighNoon model |
| `chunk_size` | `int` | `8` | Tokens per chunk |
| `tokenizer` | `Tokenizer` | `None` | Tokenizer for decoding |

### Methods

#### generate_stream()

```python
for chunk in streamer.generate_stream(prompt, **kwargs):
    # Process each chunk
    pass
```

Returns: Generator yielding text chunks

---

## Quantum Superposition Generation (QSG)

QSG enables parallel token generation for significant speedups (50-100x faster than autoregressive).

```python
from highnoon.inference import QSGGenerator

model = hn.create_model("highnoon-3b")
qsg = QSGGenerator(model)

response = qsg.generate(
    "Summarize the key points of",
    max_length=128,
    num_superposition_states=4,
)
```

### QSGGenerator

```python
from highnoon.inference import QSGGenerator

qsg = QSGGenerator(
    model,
    num_superposition_states=4,
    collapse_strategy="greedy",
    quality_threshold=0.8,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `HSMN` | *required* | HighNoon model |
| `num_superposition_states` | `int` | `4` | Parallel hypothesis states |
| `collapse_strategy` | `str` | `"greedy"` | Wavefunction collapse method |
| `quality_threshold` | `float` | `0.8` | Quality gate threshold |

### Collapse Strategies

| Strategy | Description |
|----------|-------------|
| `greedy` | Select highest probability token |
| `sample` | Sample from superposition distribution |
| `beam` | Beam search over superposition states |

> **Note**: QSG requires the native C++ operations for full performance.

---

## Speculative Decoding

Use a smaller draft model to speed up generation.

```python
from highnoon.inference import SpeculativeGenerator

# Main model (large)
target_model = hn.create_model("highnoon-7b")

# Draft model (small, fast)
draft_model = hn.create_model("highnoon-small")

generator = SpeculativeGenerator(
    target_model=target_model,
    draft_model=draft_model,
    num_speculative_tokens=4,
)

response = generator.generate(
    "Explain the theory of relativity",
    max_length=256,
)
```

### SpeculativeGenerator

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_model` | `HSMN` | *required* | Main (accurate) model |
| `draft_model` | `HSMN` | *required* | Draft (fast) model |
| `num_speculative_tokens` | `int` | `4` | Tokens to speculate per step |
| `acceptance_rate_target` | `float` | `0.7` | Target acceptance rate |

---

## Stateful Inference

Maintain state across multiple generation calls for efficient dialogue.

```python
from highnoon.inference import StatefulWrapper

model = hn.create_model("highnoon-3b")
wrapper = StatefulWrapper(model, max_history=4096)

# First turn
wrapper.add_user_message("What is machine learning?")
response1 = wrapper.generate(max_length=256)

# Second turn (uses cached context)
wrapper.add_user_message("Can you give an example?")
response2 = wrapper.generate(max_length=256)

# Reset state
wrapper.reset()
```

### StatefulWrapper

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `HSMN` | *required* | HighNoon model |
| `max_history` | `int` | `4096` | Maximum context tokens |
| `system_prompt` | `str` | `None` | System prompt for context |

### Methods

| Method | Description |
|--------|-------------|
| `add_user_message(text)` | Add user message to context |
| `add_assistant_message(text)` | Add assistant message to context |
| `generate(**kwargs)` | Generate response |
| `reset()` | Clear conversation state |
| `get_history()` | Get conversation history |

---

## Batch Inference

Process multiple prompts efficiently.

```python
model = hn.create_model("highnoon-3b")

prompts = [
    "Summarize: Machine learning is...",
    "Translate to French: Hello world",
    "Code: Write a Python function to...",
]

responses = model.generate_batch(
    prompts,
    max_length=128,
    batch_size=8,
)

for prompt, response in zip(prompts, responses):
    print(f"Prompt: {prompt[:50]}...")
    print(f"Response: {response}\n")
```

---

## Memory-Efficient Inference

For large models on limited memory:

```python
from highnoon.inference import MemoryEfficientGenerator

model = hn.create_model("highnoon-13b")

generator = MemoryEfficientGenerator(
    model,
    offload_to_disk=True,
    checkpoint_activations=True,
)

response = generator.generate(
    "Very long prompt...",
    max_length=1000,
)
```

---

## Configuration Options

Control inference behavior via configuration:

```python
from highnoon.config import Config

config = Config()

# Enable speculative decoding globally
config.USE_SPECULATIVE_DECODING = True
config.SPECULATIVE_LOOKAHEAD = 4

# Enable QSG
config.USE_QSG = True
config.QSG_NUM_SUPERPOSITION_STATES = 4

model = hn.create_model("highnoon-3b")
```

---

## Performance Comparison

| Method | Tokens/sec | Quality | Use Case |
|--------|------------|---------|----------|
| Autoregressive | ~50 | Highest | Production, accuracy-critical |
| Streaming | ~50 | Highest | Interactive applications |
| Speculative | ~150 | High | Balanced speed/quality |
| QSG | ~500-2000 | Good | Batch processing, drafts |

---

[← Training API](training.md) | [CLI API →](cli.md)
