# highnoon/__init__.py
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

"""HighNoon Language Framework - Language Intelligence, Engineered.

A lite, enterprise-grade release of the HSMN-Architecture focused on
language modeling capabilities. This framework provides the complete
architecture with enforced scale limits via tamper-proof compiled binaries.

Example Usage:
    import highnoon as hn

    # Quick inference
    model = hn.create_model("highnoon-3b")
    response = model.generate("Explain quantum computing")

    # With tool calling (Codex CLI)
    runner = hn.CodexRunner(model)
    result = runner.run("List Python files in current directory")

    # Fine-tuning with curriculum learning
    trainer = hn.Trainer(model)
    trainer.add_curriculum_stage("code_instruction", datasets=["commitpackft"])
    trainer.train(epochs=10)

For community support, visit: https://www.versoindustries.com/messages
For enterprise licensing, contact: sales@versoindustries.com
"""

import warnings

# Phase 4.2 (GRADIENT_CONNECTIVITY_ROADMAP): Suppress TensorFlow complex-to-float casting warnings.
# The FFT-based holographic operations (circular convolution, holographic binding) mathematically
# require extracting the real part from complex FFT results. This is intentional and correct.
# The warning "casting complex64/complex128 to float32" is a false positive in this context.
warnings.filterwarnings("ignore", message=".*casting.*complex.*float.*")
warnings.filterwarnings("ignore", message=".*incompatible dtype float.*imaginary part.*")

# Also filter TensorFlow's internal logging system which bypasses Python warnings
import logging


def _apply_tf_complex_cast_filter():
    """Apply filter to TensorFlow logger to suppress complex-to-float warnings."""
    try:
        tf_logger = logging.getLogger("tensorflow")

        class ComplexCastingFilter(logging.Filter):
            def filter(self, record):
                msg = record.getMessage()
                if "casting" in msg and "complex" in msg and ("float" in msg or "imaginary" in msg):
                    return False
                return True

        tf_logger.addFilter(ComplexCastingFilter())
    except Exception:
        pass  # Graceful fallback if TensorFlow not yet imported


_apply_tf_complex_cast_filter()

__version__ = "1.0.0"
__author__ = "Verso Industries"
__license__ = "Apache-2.0 (Python) + Proprietary (Binaries)"
__edition__ = "lite"


# Lazy imports to avoid circular dependencies and improve startup time
def __getattr__(name):
    """Lazy import of framework components."""
    if name == "LanguageModel":
        from highnoon.models.hsmn import HSMN

        return HSMN
    elif name == "HSMN":
        from highnoon.models.hsmn import HSMN

        return HSMN
    elif name == "create_model":
        from highnoon.models import create_model

        return create_model
    elif name == "CodexRunner":
        from highnoon.cli.runner import CodexRunner

        return CodexRunner
    elif name == "ToolManifest":
        from highnoon.cli.manifest import ToolManifest

        return ToolManifest
    elif name == "Trainer":
        from highnoon.training.trainer import Trainer

        return Trainer
    elif name == "CurriculumScheduler":
        from highnoon.training.curriculum import CurriculumScheduler

        return CurriculumScheduler
    elif name == "Config":
        from highnoon.config import Config

        return Config
    elif name == "ModelConfig":
        from highnoon.config import ModelConfig

        return ModelConfig
    elif name == "TrainingConfig":
        from highnoon.config import TrainingConfig

        return TrainingConfig
    elif name == "QWTTextTokenizer":
        from highnoon.tokenization import QWTTextTokenizer

        return QWTTextTokenizer
    elif name == "HPOTrialManager":
        from highnoon.services.hpo_manager import HPOTrialManager

        return HPOTrialManager
    # New exports for serialization and HPO bridge
    elif name == "save_model":
        from highnoon.serialization import save_model

        return save_model
    elif name == "load_model":
        from highnoon.serialization import load_model

        return load_model
    elif name == "load_tokenizer":
        from highnoon.serialization import load_tokenizer

        return load_tokenizer
    elif name == "HPOTrainingConfig":
        from highnoon.services.hpo_training_bridge import HPOTrainingConfig

        return HPOTrainingConfig
    elif name == "TrainingEngine":
        from highnoon.training.training_engine import TrainingEngine

        return TrainingEngine
    elif name == "EnterpriseTrainingConfig":
        from highnoon.training.training_engine import EnterpriseTrainingConfig

        return EnterpriseTrainingConfig
    raise AttributeError(f"module 'highnoon' has no attribute {name!r}")


__all__ = [
    "LanguageModel",
    "HSMN",
    "create_model",
    "CodexRunner",
    "ToolManifest",
    "Trainer",
    "CurriculumScheduler",
    "Config",
    "ModelConfig",
    "TrainingConfig",
    "QWTTextTokenizer",
    "HPOTrialManager",
    # Model serialization
    "save_model",
    "load_model",
    "load_tokenizer",
    # HPO-TrainingEngine integration
    "HPOTrainingConfig",
    "TrainingEngine",
    "EnterpriseTrainingConfig",
]
