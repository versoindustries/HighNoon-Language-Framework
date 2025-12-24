# highnoon/data/preprocessors/text.py
# Copyright 2025 Verso Industries
#
# Text preprocessor for raw text language modeling.

from __future__ import annotations

from typing import Any

import tensorflow as tf

from highnoon.data.preprocessors.base import BasePreprocessor


class TextPreprocessor(BasePreprocessor):
    """Preprocessor for raw text language modeling.

    Handles basic text pre-training data where each example contains
    a single text field. Suitable for:
    - Web text (CommonCrawl, C4, etc.)
    - Books and documents
    - Code pre-training

    The text is tokenized and used for causal language modeling where
    the model predicts the next token at each position.

    Example:
        >>> preprocessor = TextPreprocessor(tokenizer)
        >>> examples = [{"text": "Hello world!"}, {"text": "Another example."}]
        >>> batch = preprocessor.process(examples)
        >>> print(batch["input_ids"].shape)  # [2, max_seq_length]
    """

    def __init__(
        self,
        tokenizer: Any,
        config=None,
        text_field: str = "text",
    ):
        """Initialize text preprocessor.

        Args:
            tokenizer: Tokenizer with encode/decode methods.
            config: PreprocessorConfig instance.
            text_field: Name of the text field in examples.
        """
        super().__init__(tokenizer, config)
        self.text_field = text_field

    def process(self, examples: list[dict[str, Any]]) -> dict[str, tf.Tensor]:
        """Process text examples for language modeling.

        For causal LM, labels are shifted input_ids:
            input_ids:  [BOS, tok1, tok2, tok3, tok4]
            labels:     [tok1, tok2, tok3, tok4, EOS]

        Args:
            examples: List of dicts with 'text' field.

        Returns:
            Dict with input_ids, attention_mask, and labels tensors.
        """
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        for example in examples:
            text = example.get(self.text_field, "")

            # Tokenize
            token_ids = self.tokenize(text)

            # Add special tokens if configured
            if self.config.add_special_tokens:
                if self.config.bos_token_id is not None:
                    token_ids = [self.config.bos_token_id] + token_ids
                token_ids = token_ids + [self.config.eos_token_id]

            # Pad/truncate
            token_ids = self.pad_sequence(token_ids)

            # Create attention mask
            attention_mask = self.create_attention_mask(token_ids)

            # For causal LM: labels = input_ids shifted left
            # We use -100 for padding positions (ignored in loss)
            labels = token_ids[1:] + [self.config.pad_token_id]
            labels = [
                label if mask == 1 else -100
                for label, mask in zip(labels, attention_mask, strict=True)
            ]

            all_input_ids.append(token_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(labels)

        return {
            "input_ids": tf.constant(all_input_ids, dtype=tf.int32),
            "attention_mask": tf.constant(all_attention_masks, dtype=tf.float32),
            "labels": tf.constant(all_labels, dtype=tf.int32),
        }
