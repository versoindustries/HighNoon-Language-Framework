#!/usr/bin/env python3
# examples/quick_inference.py
# Copyright 2025 Verso Industries
#
# Quick inference example for HighNoon Language Framework.

"""Quick Inference Example.

Demonstrates basic model creation and text generation using HighNoon.

Usage:
    python examples/quick_inference.py

Requirements:
    - HighNoon Language Framework installed
    - TensorFlow 2.x
"""

import highnoon as hn


def main():
    """Run quick inference example."""
    print("HighNoon Language Framework - Quick Inference Example")
    print(f"Version: {hn.__version__}")
    print(f"Edition: {hn.__edition__}")
    print("-" * 50)

    # Create model with sensible defaults
    print("\n[1] Creating model from 'highnoon-small' preset...")
    model = hn.create_model("highnoon-small")
    print(
        f"    Model created: vocab_size={model.vocab_size}, " f"embedding_dim={model.embedding_dim}"
    )

    # For full inference with tokenizer, you would use:
    # response = model.generate(
    #     "Explain quantum computing in simple terms",
    #     max_length=512,
    #     temperature=0.7,
    #     tokenizer=tokenizer,
    # )
    # print(response)

    # Demo: Run a forward pass with dummy tokens
    print("\n[2] Running forward pass with dummy input...")
    import tensorflow as tf

    dummy_input = tf.random.uniform((1, 32), minval=0, maxval=model.vocab_size, dtype=tf.int32)
    output = model(dummy_input, training=False)
    print(f"    Input shape: {dummy_input.shape}")
    print(f"    Output shape: {output.shape}")

    # Check model config
    print("\n[3] Model configuration:")
    config = model.get_config()
    for key, value in config.items():
        print(f"    {key}: {value}")

    print("\n" + "-" * 50)
    print("Quick inference example complete!")


if __name__ == "__main__":
    main()
