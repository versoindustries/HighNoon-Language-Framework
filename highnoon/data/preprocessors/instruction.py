# highnoon/data/preprocessors/instruction.py
# Copyright 2025 Verso Industries
#
# Instruction-following preprocessor for chat/instruct fine-tuning.

from __future__ import annotations

from typing import Any

import tensorflow as tf

from highnoon.data.preprocessors.base import BasePreprocessor


class InstructionPreprocessor(BasePreprocessor):
    """Preprocessor for instruction-following fine-tuning.

    Handles instruction-response pairs for supervised fine-tuning.
    Supports various chat formats including:
    - Alpaca format (instruction, input, output)
    - ShareGPT format (conversations with roles)
    - Simple prompt-response pairs

    The loss is only computed on the response tokens (not the prompt).

    Example:
        >>> preprocessor = InstructionPreprocessor(tokenizer)
        >>> examples = [{
        ...     "instruction": "Summarize this text.",
        ...     "input": "Long text here...",
        ...     "output": "Summary here."
        ... }]
        >>> batch = preprocessor.process(examples)
    """

    # Chat template markers
    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
    INSTRUCTION_MARKER = "### Instruction:\n"
    INPUT_MARKER = "### Input:\n"
    RESPONSE_MARKER = "### Response:\n"

    def __init__(
        self,
        tokenizer: Any,
        config=None,
        format_style: str = "alpaca",
        mask_input: bool = True,
        system_prompt: str | None = None,
    ):
        """Initialize instruction preprocessor.

        Args:
            tokenizer: Tokenizer with encode/decode methods.
            config: PreprocessorConfig instance.
            format_style: Format style ('alpaca', 'simple', 'sharegpt').
            mask_input: If True, mask input in labels (only train on response).
            system_prompt: Optional system prompt to prepend.
        """
        super().__init__(tokenizer, config)
        self.format_style = format_style
        self.mask_input = mask_input
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def format_example(self, example: dict[str, Any]) -> tuple[str, str]:
        """Format an example into prompt and response strings.

        Args:
            example: Dict with instruction/input/output or prompt/response.

        Returns:
            Tuple of (prompt_text, response_text).
        """
        if self.format_style == "alpaca":
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output = example.get("output", example.get("response", ""))

            if input_text:
                prompt = (
                    f"{self.INSTRUCTION_MARKER}{instruction}\n\n"
                    f"{self.INPUT_MARKER}{input_text}\n\n"
                    f"{self.RESPONSE_MARKER}"
                )
            else:
                prompt = f"{self.INSTRUCTION_MARKER}{instruction}\n\n{self.RESPONSE_MARKER}"

            return prompt, output

        elif self.format_style == "simple":
            prompt = example.get("prompt", example.get("instruction", ""))
            response = example.get("response", example.get("output", ""))
            return prompt, response

        elif self.format_style == "sharegpt":
            # Handle conversation format
            conversations = example.get("conversations", [])
            if not conversations:
                return "", ""

            prompt_parts = []
            response = ""

            for turn in conversations:
                role = turn.get("from", turn.get("role", ""))
                value = turn.get("value", turn.get("content", ""))

                if role in ("human", "user"):
                    prompt_parts.append(f"User: {value}")
                elif role in ("gpt", "assistant"):
                    response = value  # Take last assistant turn

            prompt = "\n\n".join(prompt_parts) + "\n\nAssistant: "
            return prompt, response

        else:
            raise ValueError(f"Unknown format style: {self.format_style}")

    def process(self, examples: list[dict[str, Any]]) -> dict[str, tf.Tensor]:
        """Process instruction examples for fine-tuning.

        Args:
            examples: List of instruction-response examples.

        Returns:
            Dict with input_ids, attention_mask, and labels tensors.
            Labels are masked for prompt tokens if mask_input=True.
        """
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        for example in examples:
            prompt, response = self.format_example(example)

            # Tokenize prompt and response separately to know boundary
            prompt_ids = self.tokenize(prompt)
            response_ids = self.tokenize(response)

            # Combine with special tokens
            if self.config.add_special_tokens:
                if self.config.bos_token_id is not None:
                    token_ids = (
                        [self.config.bos_token_id]
                        + prompt_ids
                        + response_ids
                        + [self.config.eos_token_id]
                    )
                    prompt_len = len(prompt_ids) + 1  # +1 for BOS
                else:
                    token_ids = prompt_ids + response_ids + [self.config.eos_token_id]
                    prompt_len = len(prompt_ids)
            else:
                token_ids = prompt_ids + response_ids
                prompt_len = len(prompt_ids)

            # Pad/truncate
            original_len = len(token_ids)
            token_ids = self.pad_sequence(token_ids)

            # Create attention mask
            attention_mask = self.create_attention_mask(token_ids)

            # Create labels (shifted for causal LM)
            labels = token_ids[1:] + [self.config.pad_token_id]

            # Mask prompt tokens in labels if configured
            if self.mask_input:
                for i in range(min(prompt_len, len(labels))):
                    labels[i] = -100  # Ignore in loss computation

            # Mask padding tokens
            labels = [label if i < original_len else -100 for i, label in enumerate(labels)]

            all_input_ids.append(token_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(labels)

        return {
            "input_ids": tf.constant(all_input_ids, dtype=tf.int32),
            "attention_mask": tf.constant(all_attention_masks, dtype=tf.float32),
            "labels": tf.constant(all_labels, dtype=tf.int32),
        }
