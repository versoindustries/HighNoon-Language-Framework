# HD Streaming Mode Architecture Analysis

## Summary

This document addresses the HD streaming incompatibility discovered during HPO training, provides a complete unified architecture design, and defines an implementation roadmap.

**Issues Identified**:
1. **Shape Mismatch Error**: `output.shape=(64, 128, 128000)` vs `target.shape=(64,)` causing `sparse_categorical_crossentropy` to fail
2. **vocab_size Defaulting to 128000**: AdaptiveQWTTokenizer ignores QAHPO budget constraints

---

## Issue 1: HD Streaming Mode Incompatibility

### Root Cause

HD Streaming Mode (`hd_corpus.py`) and the HSMN model architecture (`build_hsmn_model`) are fundamentally incompatible:

| Component | HD Streaming Output | Model Expectation |
|-----------|---------------------|-------------------|
| **Inputs** | `float32` bundles `(batch, hd_dim)` | `int32` token IDs `(batch, seq_len)` |
| **Labels** | Single token ID `(batch,)` | Full sequence IDs `(batch, seq_len)` |
| **Model Output** | N/A | Logits `(batch, seq_len, vocab_size)` |

### What Happens

1. **HD Corpus** encodes sequences into compressed HD bundles of shape `(batch, 128)` where 128 is `hd_dim`
2. **Model** interprets the HD dim as sequence length, creating output `(batch, 128, vocab_size)`
3. **Loss function** receives `target.shape=(64,)` and `output.shape=(64, 128, 128000)` - mismatch!

### Data Flow (Current Broken Path)

```
HuggingFace Dataset
        ↓
StreamingHDTokenizer
        ↓
HolographicCorpus
        ↓
HD Bundle (batch, hd_dim=128) float32
        ↓
HSMN Model expects (batch, seq_len) int32  ← TYPE MISMATCH
        ↓
Output: (batch, 128, vocab_size)
        ↓
Loss: target=(batch,) vs output=(batch,128,vocab)  ← SHAPE MISMATCH
        ↓
❌ ERROR
```

---

## Issue 2: vocab_size Defaulting to 128000

### Root Cause

The `AdaptiveQWTTokenizer` learns n-grams from the curriculum corpus and **dynamically expands its vocabulary**, overriding any budget-aware `vocab_size` setting from QAHPO.

### Evidence from Logs

```
[AdaptiveQWT] Initialized: base_vocab=362, target=128000, codebook_capacity=127638
[AdaptiveQWT] Learned 127638 n-gram tokens, vocab_size now 128000
```

### Where 128000 Comes From

| Location | Mechanism |
|----------|-----------|
| `vocab_controller.py:142` | `target_vocab = 128000` hardcoded as "maximum practical" |
| `hpo_manager.py:126` | `min(max_vocab, 128000)` caps budget-aware calculation |
| `HPO.tsx:56` | `maxVocabSize: 128000` WebUI default |

---

## Unified Architecture Solution

### Key Insight: QSG Compatibility

QSG (Quantum Superposition Generation) performs **full-sequence parallel generation** via a 5-phase pipeline, not autoregressive single-token generation. Therefore:
- HD bundles encode the **context**, not replace sequence generation
- Training should use **full sequence labels** from HD corpus
- QSG's `_encode_context` can accept HD bundles for faster context encoding

### Unified Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING MODE                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Text ──► QWT Tokenizer ──► Token IDs [seq_len] int32                   │
│                                  │                                       │
│                                  ▼                                       │
│           ┌───────────────────────────────────────────┐                 │
│           │        HolographicCorpus                   │                 │
│           │   encode_sequence(token_ids)              │                 │
│           │       ↓                                    │                 │
│           │   HD Bundle [hd_dim] float32              │                 │
│           │   + Full sequence labels [seq_len] int32  │  ◄── NEW        │
│           └───────────────────────────────────────────┘                 │
│                                  │                                       │
│                                  ▼                                       │
│           ┌───────────────────────────────────────────┐                 │
│           │   HDStreamingAdapter Layer (NEW)          │                 │
│           │   Project: (batch, hd_dim) → (batch, H)   │                 │
│           │   Expand:  (batch, H) → (batch, S, H)     │                 │
│           └───────────────────────────────────────────┘                 │
│                                  │                                       │
│                                  ▼                                       │
│           ┌───────────────────────────────────────────┐                 │
│           │        ReasoningModule                     │                 │
│           │   (Same as standard mode)                  │                 │
│           └───────────────────────────────────────────┘                 │
│                                  │                                       │
│                                  ▼                                       │
│           ┌───────────────────────────────────────────┐                 │
│           │        QuantumLMHead                       │                 │
│           │   Output: (batch, seq, vocab) logits       │                 │
│           └───────────────────────────────────────────┘                 │
│                                  │                                       │
│                                  ▼                                       │
│           sparse_categorical_crossentropy(labels, logits)               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE MODE (QSG)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Prompt ──► QWT Tokenizer ──► Token IDs                                 │
│                                  │                                       │
│                                  ▼                                       │
│           ┌───────────────────────────────────────────┐                 │
│           │   (Optional) HD Encode for context        │                 │
│           └───────────────────────────────────────────┘                 │
│                                  │                                       │
│                                  ▼                                       │
│           ┌───────────────────────────────────────────┐                 │
│           │        QSG Generator                       │                 │
│           │   5-phase parallel sequence generation    │                 │
│           │   Full sequence output in one pass        │                 │
│           └───────────────────────────────────────────┘                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Roadmap

### Phase 1: vocab_size Budget Enforcement

> **Risk**: Low | **Effort**: 2 hours | **Priority**: FIRST

This phase is independent and affects ALL training modes.

| File | Change |
|------|--------|
| `adaptive_qwt_tokenizer.py` | Add `max_vocab_size` parameter that enforces hard cap on learned vocabulary |
| `loaders.py` | Pass `vocab_size` to AdaptiveQWTTokenizer as hard limit |
| `hpo_trial_runner.py` | Force use of `config_vocab_size` over `tokenizer.vocab_size` |
| `scheduler_factory.py` | Always populate `vocab_size` in config when `param_budget` is set |
Validate this is all tuned by the QAHPO, users do not set this parameter on their own anymore.
---

### Phase 2: HD-Aware Model Architecture

> **Risk**: Medium | **Effort**: 4 hours

| File | Change |
|------|--------|
| `hpo_trial_runner.py:build_hsmn_model` | Add `is_hd_mode: bool` parameter |
| `hpo_trial_runner.py` | Create float32 input layer when HD mode |
| `hpo_trial_runner.py` | Skip embedding layers when HD mode (bundles ARE embeddings) |
| `hpo_trial_runner.py` | Add projection: `(batch, hd_dim)` → `(batch, 1, hidden_dim)` |
| `hd_corpus.py` | Add sequence label mode to `create_dataset()` |
| **NEW** `hd_streaming_adapter.py` | Create adapter layer for HD → sequence expansion |

---

### Phase 3: QSG Integration

> **Risk**: Medium | **Effort**: 2 hours

| File | Change |
|------|--------|
| `qsg_generator.py` | Update `_encode_context` to accept HD bundles or token IDs |
| `hpo_trial_runner.py` | Pass `is_hd_mode=use_hd_streaming` when calling `build_hsmn_model` |

---

### Phase 4: QAHPO Auto-Tuning Integration

> **Risk**: Low | **Effort**: 1 hour

HD streaming is the **default architecture**—users do not toggle this unless manually editing config flags. All tunable parameters (`vocab_size`, `hd_dim`) are controlled exclusively by QAHPO.

| File | Change |
|------|--------|
| `hpo_manager.py` | Add `hd_dim` to `HPOSearchSpace` tunable parameters |
| `hpo_manager.py` | Validate `hd_dim` divisibility with `hidden_dim` |
| `hpo_trial_runner.py` | Add HD-specific logging and graceful fallback |
| `config.py` | Document `use_hd_streaming` as internal advanced flag (not WebUI-exposed) |

> [!NOTE]
> WebUI does **not** expose HD streaming toggle or vocab_size controls. These are QAHPO-managed to ensure architectural consistency and budget compliance.

---

## Verification Plan

### Automated Tests

```bash
# Phase 1: vocab_size tests
pytest tests/integration/test_hpo_sweep_webui.py::TestHPOSweepWebUI::test_vocab_size_preserved -v
pytest tests/unit/test_hpo_divisibility.py -v
pytest tests/unit/test_qahpo_enhancements.py -v

# Phase 2: HD architecture tests
pytest tests/test_hd_checkpoint_isolated.py -v
# NEW: pytest tests/unit/test_hd_streaming_model.py -v

# Phase 3: QSG integration
pytest tests/integration/test_coconut_qsg_integration.py -v

# Phase 4: Full WebUI integration
pytest tests/integration/test_hpo_sweep_webui.py -v
```

### Manual Verification

1. Start HPO sweep via WebUI with `param_budget=100M` and HD streaming enabled
2. Observe logs for:
   - `[HPO] Building HD-mode HSMN model`
   - `[HD Corpus] Creating dataset: X samples, batch_size=Y`
   - `[AdaptiveQWT] vocab_size capped at Z` (should match budget, NOT 128000)
3. Verify training completes without shape mismatch errors

---

## Implementation Order

| Order | Phase | Risk | Effort |
|-------|-------|------|--------|
| 1 | Phase 1: vocab_size Fix | Low | 2 hours |
| 2 | Phase 4: WebUI/QAHPO Prep | Low | 1 hour |
| 3 | Phase 2: HD Architecture | Medium | 4 hours |
| 4 | Phase 3: QSG Integration | Medium | 2 hours |

---

## Rollback Plan

Each phase is independent:
- **Phase 1**: Revert tokenizer changes, vocab reverts to current behavior
- **Phase 2**: Set `use_hd_streaming=False` in trial configs
- **Phase 3**: QSG continues using standard token-based context encoding
- **Phase 4**: WebUI falls back to non-HD configuration options
