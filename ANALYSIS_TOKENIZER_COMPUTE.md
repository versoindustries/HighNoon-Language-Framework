# Analysis of Tokenizer Compute Overhead (Corrected)

## Executive Summary
The investigation into the compute bottleneck associated with the tokenizer setup (`active_vocab_size: 1000`, `vocab_size: 300,000`) has identified a critical **logic reversal** in the current implementation. While the system correctly identifies 300,000 as the total vocabulary capacity (including rare/OOV tokens), the model is erroneously projecting hidden states into this 300,000-class space at the output head.

The **corrected architectural vision** is to use `active_vocab_size` (e.g., 32k) as the primary prediction space for the Dense output head, while reserving the 300,000 `vocab_size` as a system-wide cap for input-side Hyperdimensional Encoding.

## Key Findings

### 1. Architectural Logic Flip
*   **Input Layer (Correct):** `DualPathEmbedding` already implements the desired logic. It uses a dense embedding for the `active_vocab_size` (1,000–32,000) and character-level HD encoding for the remaining rare tokens up to the 300,000 `vocab_size` limit.
*   **Output Layer (Bottleneck):** `QuantumLMHead` is currently projecting to the full 300,000 `vocab_size`. This is unnecessary because rare tokens (beyond the active set) should be handled via a fallback or not predicted directly by the primary head. This projection creates the $O(V)$ bottleneck.

### 2. Physical Resource Impact
*   **Memory:** Projecting to 300,000 tokens materializes a **19.6 GB** logit tensor for a sequence length of 16,384. This exceeds the activation memory budget for standard HPO trials.
*   **Compute:** Calculating 300,000 logit values per token is unsustainable on CPU. Reducing the projection to an `active_vocab_size` of 32,000 results in a **~10x reduction** in compute and memory for the output head.

## Corrected Strategy

The model's vocabulary parameters will be standardized as follows:
1.  **`active_vocab_size` (Tuned):** The target size for the Dense prediction head and active embeddings. Typically 8,000 to 64,000.
2.  **`vocab_size` (System Cap):** The hard limit for the total number of unique tokens the system can track (including rare ones). Fixed at 300,000.

## Recommendations

1.  **Flip HPO Logic:** Standardize `active_vocab_size` as the primary search parameter in `HPOSearchSpace`. Set its upper bound to a reasonable prediction limit (e.g., 64,000).
2.  **Capped Output Projection:** Update `QuantumLMHead` and `Dense` output layers to project hidden states ONLY to the `active_vocab_size`.
3.  **HD Fallback on Input:** Maintain the 300,000 `vocab_size` cap for the `DualPathEmbedding` HD-path to ensure full coverage of the long-tail vocabulary without the compute penalty on the output side.

---
*Date: 2025-12-18*
*Status: Updated Strategy Approved*
