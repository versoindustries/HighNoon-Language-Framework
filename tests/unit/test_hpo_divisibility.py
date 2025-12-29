# tests/unit/test_hpo_divisibility.py
import numpy as np
import pytest

from highnoon.services.hpo_manager import compute_max_vocab_for_budget, estimate_model_params
from highnoon.services.hpo_utils import next_power_of_2, snap_to_multiple
from highnoon.services.quantum_hpo_scheduler import (
    QAHPOConfig,
    QuantumAdaptiveHPOScheduler,
    QuantumState,
)
from highnoon.services.sweep_executor import RetryStrategy


def test_snap_to_multiple():
    assert snap_to_multiple(358, 8) == 360
    assert snap_to_multiple(354, 8) == 352
    assert snap_to_multiple(5, 8, min_val=8) == 8
    assert snap_to_multiple(100, 32, min_val=64) == 96


def test_next_power_of_2():
    assert next_power_of_2(7) == 8
    assert next_power_of_2(8) == 8
    assert next_power_of_2(9) == 16
    assert next_power_of_2(1) == 1
    assert next_power_of_2(0) == 1


def test_qahpo_mutation_enforces_constraints():
    config = QAHPOConfig(mutation_strategy="gaussian")
    scheduler = QuantumAdaptiveHPOScheduler(config=config)

    # Mock search space sampler
    def mock_sampler(trial_id):
        return {"embedding_dim": 512, "num_heads": 8}

    scheduler.search_space_sampler = mock_sampler
    scheduler._initialize_population()

    # Mutate many times and check constraints
    base_config = {"embedding_dim": 512, "num_heads": 8}
    for _ in range(50):
        mutated = scheduler._mutate_config(base_config, strength=1.0)

        # Must be divisible by 8
        assert mutated["embedding_dim"] % 8 == 0
        assert mutated["embedding_dim"] >= 32

        # num_heads must be power of 2
        assert mutated["num_heads"] & (mutated["num_heads"] - 1) == 0


def test_retry_strategy_enforces_constraints():
    strategy = RetryStrategy()
    config = {
        "num_reasoning_blocks": 16,
        "num_moe_experts": 12,
        "hidden_dim": 1024,
        "embedding_dim": 1024,
    }

    # Test budget_exceeded retry
    for retry_count in range(1, 4):
        modified = strategy.modify_config_for_retry(config.copy(), "budget_exceeded", retry_count)

        # Dimensions must stay valid (multiples of 8)
        assert modified["hidden_dim"] % 8 == 0
        assert modified["embedding_dim"] % 8 == 0
        assert modified["hidden_dim"] >= 128

        # If num_heads was in there, it should be power of 2
        config_with_heads = config.copy()
        config_with_heads["num_heads"] = 6  # Force invalid
        modified_with_heads = strategy.modify_config_for_retry(
            config_with_heads, "budget_exceeded", retry_count
        )
        assert modified_with_heads["num_heads"] & (modified_with_heads["num_heads"] - 1) == 0


def test_compute_max_vocab_for_budget_with_hqe():
    """Test budget-aware vocab calculation with HQE enabled."""
    # With 100M budget, HQE, hidden_dim=256
    # HQE uses hd_dim = 256 * 8 = 2048
    # embed_params = vocab_size * 2048 + 2048 * 256 (projection)
    # With 50M for embeddings (50% overhead): vocab_size * 2048 = 50M - 0.5M ≈ 49.5M
    # max_vocab ≈ 49.5M / 2048 ≈ 24,169

    vocab = compute_max_vocab_for_budget(
        param_budget=100_000_000,
        embedding_dim=256,
        use_hqe=True,
        hd_dim=None,  # Will use 256*8=2048
        model_overhead_fraction=0.5,
    )

    # Should be reasonable for 100M budget
    assert 8000 <= vocab <= 30000, f"vocab={vocab} should be 8K-30K for 100M budget with HQE"


def test_compute_max_vocab_for_budget_without_hqe():
    """Test budget-aware vocab calculation without HQE."""
    # Without HQE, embed_params = vocab_size * embedding_dim
    # With 50M for embeddings, hidden_dim=256: vocab_size = 50M / 256 ≈ 195,312
    # Capped at 128K

    vocab = compute_max_vocab_for_budget(
        param_budget=100_000_000,
        embedding_dim=256,
        use_hqe=False,
        model_overhead_fraction=0.5,
    )

    # Without HQE, can support much larger vocab
    assert vocab == 128000, f"vocab={vocab} should be capped at 128K without HQE"


def test_compute_max_vocab_with_explicit_hd_dim():
    """Test budget-aware vocab with explicit hd_dim (smaller than default)."""
    vocab = compute_max_vocab_for_budget(
        param_budget=100_000_000,
        embedding_dim=256,
        use_hqe=True,
        hd_dim=512,  # Much smaller than default 2048
        model_overhead_fraction=0.5,
    )

    # With smaller hd_dim, can support larger vocab
    assert vocab > 50000, f"vocab={vocab} should be >50K with small hd_dim=512"


def test_estimate_model_params_with_budget():
    """Test that estimate_model_params uses budget-aware vocab when vocab_size is None."""
    config = {
        "vocab_size": None,  # Triggers budget-aware calculation
        "param_budget": 100_000_000,
        "hidden_dim": 256,
        "use_hyperdimensional_embedding": True,
        "num_reasoning_blocks": 4,
        "num_moe_experts": 4,
    }

    estimated = estimate_model_params(config)

    # With budget-aware vocab, params should be under budget
    assert estimated < 150_000_000, f"Estimated params {estimated/1e6:.1f}M should be near 100M"


def test_estimate_model_params_respects_explicit_vocab():
    """Test that estimate_model_params respects explicit vocab_size."""
    config = {
        "vocab_size": 16000,  # Explicit
        "param_budget": 100_000_000,
        "hidden_dim": 256,
        "use_hyperdimensional_embedding": True,
        "num_reasoning_blocks": 4,
        "num_moe_experts": 4,
    }

    estimated = estimate_model_params(config)

    # With explicit small vocab, should be well under budget
    assert estimated < 100_000_000, f"Estimated {estimated/1e6:.1f}M should be under 100M"


def test_retry_strategy_sets_vocab_on_budget_exceeded():
    """Test that retry strategy sets vocab_size when it's None and budget is exceeded."""
    strategy = RetryStrategy()
    config = {
        "vocab_size": None,  # Will be set by retry
        "param_budget": 100_000_000,
        "num_reasoning_blocks": 4,
        "num_moe_experts": 4,
        "hidden_dim": 256,
        "use_hyperdimensional_embedding": True,
    }

    modified = strategy.modify_config_for_retry(config.copy(), "budget_exceeded", 1)

    # Should now have explicit vocab_size
    assert (
        modified.get("vocab_size") is not None
    ), "vocab_size should be set on budget_exceeded retry"
    assert modified["vocab_size"] > 0, f"vocab_size should be positive: {modified['vocab_size']}"
    assert modified["vocab_size"] < 50000, "vocab_size should be <50K for 100M budget with HQE"


def test_retry_strategy_keeps_hqe_enabled():
    """Test that retry strategy keeps HQE enabled (budget-aware vocab should suffice)."""
    strategy = RetryStrategy()
    config = {
        "vocab_size": 16000,
        "param_budget": 100_000_000,
        "num_reasoning_blocks": 4,
        "num_moe_experts": 4,
        "hidden_dim": 256,
        "use_hyperdimensional_embedding": True,
    }

    # HQE should remain enabled across all retries
    for retry_count in [1, 2, 3]:
        modified = strategy.modify_config_for_retry(config.copy(), "budget_exceeded", retry_count)
        assert (
            modified.get("use_hyperdimensional_embedding", True) is True
        ), f"HQE should stay enabled on retry {retry_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
