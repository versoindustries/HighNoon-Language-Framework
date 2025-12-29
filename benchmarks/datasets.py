# benchmarks/datasets.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified HuggingFace dataset loading and preprocessing for benchmarks.

Provides streaming data loaders for memory-efficient evaluation on
large datasets including WikiText, LongBench, SCROLLS, and more.

Example:
    >>> from benchmarks.datasets import DatasetRegistry
    >>> registry = DatasetRegistry()
    >>> loader = registry.get_loader("wikitext-2", max_samples=1000)
    >>> for text in loader:
    ...     print(len(text))

Command-line usage:
    python -m benchmarks.datasets --dataset wikitext-2 --info
"""

import logging
import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a single dataset.

    Attributes:
        name: Short name for the dataset.
        hf_path: HuggingFace dataset path.
        hf_name: HuggingFace dataset configuration name.
        split: Dataset split to use.
        text_column: Column containing text data.
        description: Human-readable description.
        streaming: Whether to use streaming mode (recommended for large).
        max_length: Maximum context length this dataset is suitable for.
        category: Dataset category (perplexity, long_context, reasoning, etc).
    """

    name: str
    hf_path: str
    hf_name: str | None = None
    split: str = "test"
    text_column: str = "text"
    description: str = ""
    streaming: bool = True
    max_length: int = 4096
    category: str = "perplexity"


# Standard benchmark datasets registry
DATASET_REGISTRY: dict[str, DatasetConfig] = {
    # Perplexity benchmarks
    "wikitext-2": DatasetConfig(
        name="wikitext-2",
        hf_path="wikitext",
        hf_name="wikitext-2-raw-v1",
        split="test",
        text_column="text",
        description="WikiText-2 test set - standard language modeling benchmark",
        max_length=4096,
        category="perplexity",
    ),
    "wikitext-103": DatasetConfig(
        name="wikitext-103",
        hf_path="wikitext",
        hf_name="wikitext-103-raw-v1",
        split="test",
        text_column="text",
        description="WikiText-103 test set - larger language modeling benchmark",
        max_length=8192,
        category="perplexity",
    ),
    "ptb": DatasetConfig(
        name="ptb",
        hf_path="ptb_text_only",
        hf_name=None,
        split="test",
        text_column="sentence",
        description="Penn Treebank - classic language modeling benchmark",
        max_length=2048,
        category="perplexity",
    ),
    "c4": DatasetConfig(
        name="c4",
        hf_path="allenai/c4",
        hf_name="en",
        split="validation",
        text_column="text",
        description="C4 validation set - web text benchmark",
        max_length=8192,
        category="perplexity",
    ),
    "the-pile": DatasetConfig(
        name="the-pile",
        hf_path="EleutherAI/pile",
        hf_name=None,
        split="test",
        text_column="text",
        description="The Pile - diverse text corpus",
        max_length=8192,
        category="perplexity",
    ),
    # Long context benchmarks
    "longbench-narrativeqa": DatasetConfig(
        name="longbench-narrativeqa",
        hf_path="THUDM/LongBench",
        hf_name="narrativeqa",
        split="test",
        text_column="context",
        description="LongBench NarrativeQA - long document QA",
        max_length=128000,
        category="long_context",
    ),
    "longbench-qasper": DatasetConfig(
        name="longbench-qasper",
        hf_path="THUDM/LongBench",
        hf_name="qasper",
        split="test",
        text_column="context",
        description="LongBench Qasper - scientific paper QA",
        max_length=128000,
        category="long_context",
    ),
    "longbench-multifieldqa": DatasetConfig(
        name="longbench-multifieldqa",
        hf_path="THUDM/LongBench",
        hf_name="multifieldqa_en",
        split="test",
        text_column="context",
        description="LongBench Multi-field QA - diverse domain QA",
        max_length=128000,
        category="long_context",
    ),
    "longbench-hotpotqa": DatasetConfig(
        name="longbench-hotpotqa",
        hf_path="THUDM/LongBench",
        hf_name="hotpotqa",
        split="test",
        text_column="context",
        description="LongBench HotpotQA - multi-hop reasoning",
        max_length=64000,
        category="long_context",
    ),
    "longbench-2wikimqa": DatasetConfig(
        name="longbench-2wikimqa",
        hf_path="THUDM/LongBench",
        hf_name="2wikimqa",
        split="test",
        text_column="context",
        description="LongBench 2WikiMQA - multi-document QA",
        max_length=64000,
        category="long_context",
    ),
    "longbench-musique": DatasetConfig(
        name="longbench-musique",
        hf_path="THUDM/LongBench",
        hf_name="musique",
        split="test",
        text_column="context",
        description="LongBench MuSiQue - multi-hop reasoning",
        max_length=64000,
        category="long_context",
    ),
    "longbench-gov-report": DatasetConfig(
        name="longbench-gov-report",
        hf_path="THUDM/LongBench",
        hf_name="gov_report",
        split="test",
        text_column="context",
        description="LongBench GovReport - long document summarization",
        max_length=128000,
        category="long_context",
    ),
    "longbench-qmsum": DatasetConfig(
        name="longbench-qmsum",
        hf_path="THUDM/LongBench",
        hf_name="qmsum",
        split="test",
        text_column="context",
        description="LongBench QMSum - meeting summarization",
        max_length=64000,
        category="long_context",
    ),
    "longbench-multi-news": DatasetConfig(
        name="longbench-multi-news",
        hf_path="THUDM/LongBench",
        hf_name="multi_news",
        split="test",
        text_column="context",
        description="LongBench MultiNews - multi-document summarization",
        max_length=64000,
        category="long_context",
    ),
    "longbench-triviaqa": DatasetConfig(
        name="longbench-triviaqa",
        hf_path="THUDM/LongBench",
        hf_name="triviaqa",
        split="test",
        text_column="context",
        description="LongBench TriviaQA - factual retrieval",
        max_length=128000,
        category="long_context",
    ),
    "longbench-samsum": DatasetConfig(
        name="longbench-samsum",
        hf_path="THUDM/LongBench",
        hf_name="samsum",
        split="test",
        text_column="context",
        description="LongBench SAMSum - dialogue summarization",
        max_length=16000,
        category="long_context",
    ),
    "longbench-passage-count": DatasetConfig(
        name="longbench-passage-count",
        hf_path="THUDM/LongBench",
        hf_name="passage_count",
        split="test",
        text_column="context",
        description="LongBench Passage Count - counting synthetic task",
        max_length=128000,
        category="long_context",
    ),
    "longbench-passage-retrieval": DatasetConfig(
        name="longbench-passage-retrieval",
        hf_path="THUDM/LongBench",
        hf_name="passage_retrieval_en",
        split="test",
        text_column="context",
        description="LongBench Passage Retrieval - retrieval synthetic task",
        max_length=128000,
        category="long_context",
    ),
    # SCROLLS dataset
    "scrolls-qasper": DatasetConfig(
        name="scrolls-qasper",
        hf_path="tau/scrolls",
        hf_name="qasper",
        split="test",
        text_column="input",
        description="SCROLLS Qasper - scientific paper QA",
        max_length=64000,
        category="long_context",
    ),
    "scrolls-narrative-qa": DatasetConfig(
        name="scrolls-narrative-qa",
        hf_path="tau/scrolls",
        hf_name="narrative_qa",
        split="test",
        text_column="input",
        description="SCROLLS NarrativeQA - story understanding",
        max_length=128000,
        category="long_context",
    ),
    "scrolls-gov-report": DatasetConfig(
        name="scrolls-gov-report",
        hf_path="tau/scrolls",
        hf_name="gov_report",
        split="test",
        text_column="input",
        description="SCROLLS GovReport - long document summarization",
        max_length=128000,
        category="long_context",
    ),
    "scrolls-summ-screen-fd": DatasetConfig(
        name="scrolls-summ-screen-fd",
        hf_path="tau/scrolls",
        hf_name="summ_screen_fd",
        split="test",
        text_column="input",
        description="SCROLLS SummScreenFD - TV script summarization",
        max_length=64000,
        category="long_context",
    ),
    "scrolls-qmsum": DatasetConfig(
        name="scrolls-qmsum",
        hf_path="tau/scrolls",
        hf_name="qmsum",
        split="test",
        text_column="input",
        description="SCROLLS QMSum - meeting summarization",
        max_length=64000,
        category="long_context",
    ),
    "scrolls-squality": DatasetConfig(
        name="scrolls-squality",
        hf_path="tau/scrolls",
        hf_name="squality",
        split="test",
        text_column="input",
        description="SCROLLS SQuALITY - story summarization",
        max_length=64000,
        category="long_context",
    ),
    "scrolls-contract-nli": DatasetConfig(
        name="scrolls-contract-nli",
        hf_path="tau/scrolls",
        hf_name="contract_nli",
        split="test",
        text_column="input",
        description="SCROLLS ContractNLI - legal document NLI",
        max_length=16000,
        category="long_context",
    ),
    # Reasoning benchmarks (for trained models)
    "arc-challenge": DatasetConfig(
        name="arc-challenge",
        hf_path="ai2_arc",
        hf_name="ARC-Challenge",
        split="test",
        text_column="question",
        description="ARC-Challenge - grade school science reasoning",
        max_length=512,
        category="reasoning",
    ),
    "arc-easy": DatasetConfig(
        name="arc-easy",
        hf_path="ai2_arc",
        hf_name="ARC-Easy",
        split="test",
        text_column="question",
        description="ARC-Easy - easier science questions",
        max_length=512,
        category="reasoning",
    ),
    "hellaswag": DatasetConfig(
        name="hellaswag",
        hf_path="hellaswag",
        hf_name=None,
        split="validation",
        text_column="ctx",
        description="HellaSwag - commonsense reasoning",
        max_length=512,
        category="reasoning",
    ),
    "winogrande": DatasetConfig(
        name="winogrande",
        hf_path="winogrande",
        hf_name="winogrande_xl",
        split="validation",
        text_column="sentence",
        description="WinoGrande - pronoun resolution",
        max_length=256,
        category="reasoning",
    ),
    # Knowledge benchmarks (for trained models)
    "mmlu": DatasetConfig(
        name="mmlu",
        hf_path="cais/mmlu",
        hf_name="all",
        split="test",
        text_column="question",
        description="MMLU - massive multitask language understanding",
        max_length=1024,
        category="knowledge",
    ),
    "truthfulqa": DatasetConfig(
        name="truthfulqa",
        hf_path="truthful_qa",
        hf_name="multiple_choice",
        split="validation",
        text_column="question",
        description="TruthfulQA - factual accuracy",
        max_length=512,
        category="knowledge",
    ),
    # Math benchmarks (for trained models)
    "gsm8k": DatasetConfig(
        name="gsm8k",
        hf_path="gsm8k",
        hf_name="main",
        split="test",
        text_column="question",
        description="GSM8K - grade school math reasoning",
        max_length=1024,
        category="math",
    ),
    # Code benchmarks (for trained models)
    "humaneval": DatasetConfig(
        name="humaneval",
        hf_path="openai_humaneval",
        hf_name=None,
        split="test",
        text_column="prompt",
        description="HumanEval - Python code generation",
        max_length=2048,
        category="code",
    ),
    "mbpp": DatasetConfig(
        name="mbpp",
        hf_path="mbpp",
        hf_name=None,
        split="test",
        text_column="text",
        description="MBPP - Python programming benchmark",
        max_length=1024,
        category="code",
    ),
}


class DatasetRegistry:
    """Registry and loader for benchmark datasets.

    Provides unified interface for loading various HuggingFace datasets
    with streaming support for memory efficiency.

    Example:
        >>> registry = DatasetRegistry()
        >>> print(registry.list_datasets("long_context"))
        ['longbench-narrativeqa', 'longbench-qasper', ...]
    """

    def __init__(self) -> None:
        """Initialize dataset registry."""
        self._cache: dict[str, Any] = {}

    def list_datasets(self, category: str | None = None) -> list[str]:
        """List available datasets.

        Args:
            category: Filter by category (perplexity, long_context, reasoning, etc).

        Returns:
            List of dataset names.
        """
        if category is None:
            return list(DATASET_REGISTRY.keys())
        return [name for name, config in DATASET_REGISTRY.items() if config.category == category]

    def get_config(self, name: str) -> DatasetConfig:
        """Get dataset configuration.

        Args:
            name: Dataset name.

        Returns:
            DatasetConfig for the dataset.

        Raises:
            ValueError: If dataset not found.
        """
        if name not in DATASET_REGISTRY:
            available = ", ".join(DATASET_REGISTRY.keys())
            raise ValueError(f"Unknown dataset: {name}. Available: {available}")
        return DATASET_REGISTRY[name]

    def get_info(self, name: str) -> dict[str, Any]:
        """Get dataset information.

        Args:
            name: Dataset name.

        Returns:
            Dictionary with dataset info.
        """
        config = self.get_config(name)
        return {
            "name": config.name,
            "hf_path": config.hf_path,
            "hf_name": config.hf_name,
            "split": config.split,
            "description": config.description,
            "category": config.category,
            "max_length": config.max_length,
        }

    def load_dataset(
        self,
        name: str,
        max_samples: int | None = None,
        streaming: bool | None = None,
    ) -> Iterator[str]:
        """Load dataset as text iterator.

        Args:
            name: Dataset name from registry.
            max_samples: Maximum samples to yield.
            streaming: Override streaming setting.

        Yields:
            Text strings from the dataset.

        Raises:
            ImportError: If datasets library not installed.
            ValueError: If dataset not found.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library required. Install with: pip install datasets"
            ) from None

        config = self.get_config(name)
        use_streaming = streaming if streaming is not None else config.streaming

        logger.info(f"Loading {name}: {config.description}")

        try:
            dataset = load_dataset(
                config.hf_path,
                config.hf_name,
                split=config.split,
                streaming=use_streaming,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
            raise

        yielded = 0
        min_length = 50  # Skip very short texts

        for item in dataset:
            text = item.get(config.text_column, "")
            if text and len(text) > min_length:
                yield text
                yielded += 1
                if max_samples is not None and yielded >= max_samples:
                    break

        logger.info(f"Loaded {yielded} samples from {name}")

    def load_tokenized_batches(
        self,
        name: str,
        tokenizer,
        batch_size: int = 8,
        max_length: int = 512,
        max_samples: int | None = None,
    ) -> Iterator[tuple]:
        """Load dataset as tokenized batches.

        Args:
            name: Dataset name.
            tokenizer: Tokenizer with __call__ method.
            batch_size: Batch size.
            max_length: Maximum sequence length.
            max_samples: Maximum samples to process.

        Yields:
            Tuples of (input_ids, target_ids) tensors.
        """
        import tensorflow as tf

        batch_inputs = []
        batch_targets = []

        for text in self.load_dataset(name, max_samples):
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            tokens = np.array(encoded["input_ids"], dtype=np.int32)
            target = np.roll(tokens, -1)

            batch_inputs.append(tokens)
            batch_targets.append(target)

            if len(batch_inputs) >= batch_size:
                yield (
                    tf.constant(np.array(batch_inputs), dtype=tf.int32),
                    tf.constant(np.array(batch_targets), dtype=tf.int32),
                )
                batch_inputs = []
                batch_targets = []

        # Yield remaining
        if batch_inputs:
            yield (
                tf.constant(np.array(batch_inputs), dtype=tf.int32),
                tf.constant(np.array(batch_targets), dtype=tf.int32),
            )


class SyntheticDataGenerator:
    """Generate synthetic text data for benchmarking.

    Provides deterministic synthetic text generation for testing
    without requiring external datasets.

    Example:
        >>> gen = SyntheticDataGenerator(seed=42)
        >>> text = gen.generate_haystack(100000)  # 100K chars
        >>> needle = gen.generate_needle()
    """

    # Sample vocabulary for synthetic text
    VOCAB = [
        "the",
        "a",
        "is",
        "was",
        "were",
        "are",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "and",
        "but",
        "or",
        "nor",
        "for",
        "yet",
        "so",
        "after",
        "although",
        "because",
        "before",
        "if",
        "since",
        "though",
        "unless",
        "until",
        "when",
        "where",
        "while",
        "that",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "how",
        "why",
        "in",
        "on",
        "at",
        "by",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "from",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "also",
        "now",
        "new",
        "old",
        "high",
        "low",
        "good",
        "bad",
        "great",
        "small",
        "large",
        "long",
        "short",
        "right",
        "wrong",
        "different",
        "important",
        "significant",
        "major",
        "minor",
        "key",
        "main",
        "primary",
    ]

    TOPICS = [
        "technology",
        "science",
        "history",
        "literature",
        "mathematics",
        "philosophy",
        "economics",
        "psychology",
        "biology",
        "physics",
        "chemistry",
        "astronomy",
        "geography",
        "politics",
        "sociology",
    ]

    def __init__(self, seed: int = 42) -> None:
        """Initialize generator with seed for reproducibility."""
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def generate_sentence(self, min_words: int = 8, max_words: int = 25) -> str:
        """Generate a random sentence.

        Args:
            min_words: Minimum words per sentence.
            max_words: Maximum words per sentence.

        Returns:
            Random sentence string.
        """
        num_words = self.rng.randint(min_words, max_words)
        words = [self.rng.choice(self.VOCAB) for _ in range(num_words)]
        words[0] = words[0].capitalize()
        return " ".join(words) + "."

    def generate_paragraph(self, min_sentences: int = 3, max_sentences: int = 8) -> str:
        """Generate a random paragraph.

        Args:
            min_sentences: Minimum sentences per paragraph.
            max_sentences: Maximum sentences per paragraph.

        Returns:
            Random paragraph string.
        """
        num_sentences = self.rng.randint(min_sentences, max_sentences)
        sentences = [self.generate_sentence() for _ in range(num_sentences)]
        return " ".join(sentences)

    def generate_haystack(
        self,
        target_chars: int,
        topic: str | None = None,
    ) -> str:
        """Generate a large text block (haystack) for needle-in-haystack tests.

        Args:
            target_chars: Target character count.
            topic: Optional topic to include periodically.

        Returns:
            Large text block.
        """
        topic = topic or self.rng.choice(self.TOPICS)
        paragraphs = []
        total_chars = 0

        while total_chars < target_chars:
            # Occasionally mention topic to make text more coherent
            if self.rng.random() < 0.1:
                para = f"This section discusses {topic}. " + self.generate_paragraph()
            else:
                para = self.generate_paragraph()

            paragraphs.append(para)
            total_chars += len(para) + 2  # +2 for newlines

        return "\n\n".join(paragraphs)

    def generate_needle(self, needle_id: int = 1) -> tuple[str, str]:
        """Generate a fact to hide (needle) and its retrieval question.

        Args:
            needle_id: Unique identifier for this needle.

        Returns:
            Tuple of (needle_text, question).
        """
        facts = [
            (
                "The secret passphrase is 'quantum-butterfly-7849'.",
                "What is the secret passphrase?",
            ),
            ("The hidden number you need to remember is 42.", "What is the hidden number?"),
            (
                "The answer to the ultimate question is 'forty-two'.",
                "What is the answer to the ultimate question?",
            ),
            ("The code for the vault is 314159.", "What is the code for the vault?"),
            (
                "The name of the secret agent is 'Nighthawk'.",
                "What is the name of the secret agent?",
            ),
            (
                "The coordinates of the treasure are 37.7749, -122.4194.",
                "What are the coordinates of the treasure?",
            ),
            (
                "The password for the database is 'entropy-cascade'.",
                "What is the password for the database?",
            ),
            ("The special ingredient is saffron from Kashmir.", "What is the special ingredient?"),
        ]

        idx = needle_id % len(facts)
        return facts[idx]

    def generate_passkey(self) -> tuple[str, str, str]:
        """Generate a passkey retrieval test.

        Returns:
            Tuple of (instruction, passkey, expected_answer).
        """
        passkey = "".join([str(self.rng.randint(0, 9)) for _ in range(5)])

        instruction = (
            f"The passkey is {passkey}. Remember this passkey. " "You will be asked for it later."
        )

        return instruction, passkey, passkey

    def generate_multi_needle_test(
        self,
        haystack_chars: int,
        num_needles: int = 3,
        positions: list[float] | None = None,
    ) -> dict[str, Any]:
        """Generate a multi-needle test case.

        Args:
            haystack_chars: Size of haystack in characters.
            num_needles: Number of needles to insert.
            positions: Relative positions (0-1) for needles.

        Returns:
            Dictionary with test data.
        """
        if positions is None:
            positions = [0.1, 0.5, 0.9][:num_needles]

        haystack = self.generate_haystack(haystack_chars)
        paragraphs = haystack.split("\n\n")
        total_paras = len(paragraphs)

        needles = []
        questions = []

        for i, pos in enumerate(positions[:num_needles]):
            needle, question = self.generate_needle(i + 1)
            needles.append(needle)
            questions.append(question)

            # Insert needle at relative position
            insert_idx = int(pos * total_paras)
            insert_idx = max(0, min(insert_idx, total_paras - 1))
            paragraphs[insert_idx] = paragraphs[insert_idx] + " " + needle

        full_text = "\n\n".join(paragraphs)

        return {
            "text": full_text,
            "needles": needles,
            "questions": questions,
            "positions": positions,
            "haystack_chars": len(full_text),
            "num_paragraphs": len(paragraphs),
        }


def main() -> int:
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="HSMN Benchmark Datasets")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        help="Dataset to inspect or load",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show dataset information",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets",
    )
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        choices=["perplexity", "long_context", "reasoning", "knowledge", "math", "code"],
        help="Filter by category",
    )
    parser.add_argument(
        "--sample",
        "-s",
        type=int,
        default=0,
        help="Number of samples to display",
    )

    args = parser.parse_args()

    registry = DatasetRegistry()

    if args.list:
        datasets = registry.list_datasets(args.category)
        print(f"Available datasets ({len(datasets)} total):")
        print()
        for name in datasets:
            config = registry.get_config(name)
            print(f"  {name:30} [{config.category}]")
            print(f"    {config.description}")
        return 0

    if args.dataset:
        if args.info:
            info = registry.get_info(args.dataset)
            print(f"Dataset: {info['name']}")
            print(f"  HuggingFace Path: {info['hf_path']}")
            print(f"  Config: {info['hf_name']}")
            print(f"  Split: {info['split']}")
            print(f"  Category: {info['category']}")
            print(f"  Max Length: {info['max_length']:,} tokens")
            print(f"  Description: {info['description']}")
            return 0

        if args.sample > 0:
            print(f"Loading {args.sample} samples from {args.dataset}...")
            for i, text in enumerate(registry.load_dataset(args.dataset, args.sample)):
                print(f"\n--- Sample {i+1} ({len(text)} chars) ---")
                print(text[:500] + ("..." if len(text) > 500 else ""))
            return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
