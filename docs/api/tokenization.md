# Tokenization API Reference

> HighNoon Language Framework - Tokenizers

This document covers the tokenization system including `QWTTextTokenizer`, `ByteStreamTokenizer`, and `SuperwordMerger`.

---

## Overview

HighNoon provides several tokenization approaches:

| Tokenizer | Description | Use Case |
|-----------|-------------|----------|
| `QWTTextTokenizer` | Quantum Wavelet Transform tokenizer | Primary tokenizer |
| `ByteStreamTokenizer` | Raw byte-level tokenizer | Low-level processing |
| `SuperwordMerger` | BPE-style subword merging | Vocabulary compression |

---

## QWTTextTokenizer

The primary tokenizer for HighNoon models, implementing Quantum Wavelet Transform encoding.

```python
from highnoon.tokenization import QWTTextTokenizer

tokenizer = QWTTextTokenizer(
    vocab_size=32000,
    model_path=None,
)
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | `int` | `32000` | Target vocabulary size |
| `model_path` | `str` | `None` | Path to pre-trained tokenizer |

### Methods

#### encode()

Encode text to token IDs.

```python
tokens = tokenizer.encode("Hello, world!")
# [101, 7592, 1010, 2088, 999, 102]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Input text |
| `add_special_tokens` | `bool` | `True` | Add BOS/EOS tokens |
| `max_length` | `int` | `None` | Truncate to max length |
| `padding` | `bool` | `False` | Pad to max_length |

Returns: `list[int]` - Token IDs

#### decode()

Decode token IDs to text.

```python
text = tokenizer.decode([101, 7592, 1010, 2088, 999, 102])
# "Hello, world!"
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `token_ids` | `list[int]` | Token IDs to decode |
| `skip_special_tokens` | `bool` | Skip special tokens |

Returns: `str` - Decoded text

#### batch_encode()

Encode multiple texts.

```python
batch_tokens = tokenizer.batch_encode([
    "First text",
    "Second text",
    "Third text",
], padding=True, max_length=128)
```

Returns: `list[list[int]]` - Batch of token ID sequences

#### batch_decode()

Decode multiple token sequences.

```python
texts = tokenizer.batch_decode(batch_tokens)
```

Returns: `list[str]` - Decoded texts

#### train()

Train the tokenizer on a corpus.

```python
tokenizer.train(
    files=["corpus.txt"],
    vocab_size=32000,
    min_frequency=2,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `files` | `list[str]` | Training corpus files |
| `vocab_size` | `int` | Target vocabulary size |
| `min_frequency` | `int` | Minimum token frequency |

#### save()

Save tokenizer to disk.

```python
tokenizer.save("./my_tokenizer")
```

#### load() (classmethod)

Load tokenizer from disk.

```python
tokenizer = QWTTextTokenizer.load("./my_tokenizer")
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `vocab_size` | `int` | Vocabulary size |
| `pad_token_id` | `int` | Padding token ID |
| `bos_token_id` | `int` | Beginning of sequence ID |
| `eos_token_id` | `int` | End of sequence ID |
| `unk_token_id` | `int` | Unknown token ID |

---

## ByteStreamTokenizer

Low-level byte tokenizer for processing raw bytes.

```python
from highnoon.tokenization import ByteStreamTokenizer

tokenizer = ByteStreamTokenizer()
```

### Methods

#### encode()

```python
tokens = tokenizer.encode("Hello")
# [72, 101, 108, 108, 111]  # ASCII values
```

#### decode()

```python
text = tokenizer.decode([72, 101, 108, 108, 111])
# "Hello"
```

#### encode_bytes()

Encode raw bytes.

```python
tokens = tokenizer.encode_bytes(b"\x00\x01\x02")
# [0, 1, 2]
```

---

## SuperwordMerger

BPE-style subword merging for vocabulary compression.

```python
from highnoon.tokenization import SuperwordMerger

merger = SuperwordMerger(
    num_merges=1000,
    min_frequency=2,
)
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_merges` | `int` | `1000` | Number of BPE merges |
| `min_frequency` | `int` | `2` | Minimum pair frequency |

### Methods

#### learn_merges()

Learn merge rules from corpus.

```python
merger.learn_merges(corpus=["text1", "text2", ...])
```

#### apply_merges()

Apply learned merges to tokens.

```python
merged = merger.apply_merges(tokens)
```

#### get_merge_rules()

Get learned merge rules.

```python
rules = merger.get_merge_rules()
# [("l", "o") -> "lo", ("lo", "w") -> "low", ...]
```

---

## Integration with Models

### Using Tokenizer with Generation

```python
import highnoon as hn
from highnoon.tokenization import QWTTextTokenizer

# Load model and tokenizer
model = hn.create_model("highnoon-3b")
tokenizer = QWTTextTokenizer.load("./tokenizer")

# Encode prompt
prompt = "Explain machine learning:"
input_ids = tokenizer.encode(prompt, add_special_tokens=True)

# Generate
import tensorflow as tf
input_tensor = tf.constant([input_ids])
output = model.generate(input_tensor, max_length=256)

# Decode
response = tokenizer.decode(output[0].numpy().tolist())
print(response)
```

### Using Tokenizer with Training

```python
import highnoon as hn
from highnoon.tokenization import QWTTextTokenizer

model = hn.create_model("highnoon-small")
tokenizer = QWTTextTokenizer(vocab_size=32000)

# Train tokenizer on your data
tokenizer.train(
    files=["train_corpus.txt"],
    vocab_size=32000,
)

# Save for later use
tokenizer.save("./my_tokenizer")

# Use with trainer
trainer = hn.Trainer(model)
# Tokenizer is used internally for data processing
```

---

## Special Tokens

| Token | ID (default) | Purpose |
|-------|--------------|---------|
| `[PAD]` | 0 | Padding |
| `[UNK]` | 1 | Unknown token |
| `[BOS]` | 2 | Beginning of sequence |
| `[EOS]` | 3 | End of sequence |
| `[SEP]` | 4 | Separator |
| `[CLS]` | 5 | Classification token |
| `[MASK]` | 6 | Mask token (for MLM) |

```python
# Access special tokens
print(tokenizer.pad_token_id)  # 0
print(tokenizer.bos_token_id)  # 2
print(tokenizer.eos_token_id)  # 3
```

---

## Complete Example

```python
from highnoon.tokenization import QWTTextTokenizer, SuperwordMerger

# Create and train tokenizer
tokenizer = QWTTextTokenizer(vocab_size=50000)

# Train on corpus
tokenizer.train(
    files=[
        "data/wikipedia.txt",
        "data/books.txt",
        "data/code.txt",
    ],
    vocab_size=50000,
    min_frequency=3,
)

# Save tokenizer
tokenizer.save("./production_tokenizer")

# Test encoding/decoding
text = "The quick brown fox jumps over the lazy dog."
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)

print(f"Original: {text}")
print(f"Tokens: {tokens}")
print(f"Decoded: {decoded}")
print(f"Vocab size: {tokenizer.vocab_size}")

# Batch processing
texts = [
    "First example text.",
    "Second example with more words.",
    "Short.",
]

batch = tokenizer.batch_encode(
    texts,
    padding=True,
    max_length=64,
)

for text, tokens in zip(texts, batch):
    print(f"{text[:30]:30s} -> {len(tokens)} tokens")
```

---

[← Config API](config.md) | [Training Guide →](../guides/training.md)
