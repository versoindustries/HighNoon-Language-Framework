# HighNoon Language Framework — Lite Edition

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/tensorflow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![Platform](https://img.shields.io/badge/platform-Linux%20x86__64-lightgrey.svg)]()
[![Status](https://img.shields.io/badge/status-Active%20Development-brightgreen.svg)]()

<p align="center">
  <img src="docs/images/highnoon_logo.svg" alt="HighNoon Logo" width="300"/>
</p>

> [!IMPORTANT]
> **Platform Support:** Native C++ operations are currently compiled for **Linux x86_64 only**. macOS and Windows support is planned for future releases. Python fallback operations are available for development and testing on other platforms.

> [!NOTE]
> **Active Development:** This framework is under active development with frequent updates. Star and watch this repository to stay notified of new features, optimizations, and releases. More commits coming soon!

---

## 🔗 Community & Resources

<p align="center">
  <a href="https://huggingface.co/versoindustries"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Models-yellow.svg" alt="HuggingFace"/></a>
  <a href="https://github.com/versoindustries/HighNoon-Language-Framework"><img src="https://img.shields.io/badge/GitHub-Repository-181717.svg?logo=github" alt="GitHub"/></a>
  <a href="https://versoindustries.com"><img src="https://img.shields.io/badge/Verso-Industries-blue.svg" alt="Verso Industries"/></a>
</p>

- **🤗 HuggingFace**: [huggingface.co/versoindustries](https://huggingface.co/versoindustries) — Pre-trained models, datasets, and spaces
- **📦 PyPI**: Coming soon — `pip install highnoon`
- **📖 Documentation**: [docs/](docs/) — API references and guides
- **💬 Community**: [versoindustries.com/messages](https://www.versoindustries.com/messages) — Questions, feedback, and discussions

---

**HighNoon** is the **Lite Edition** of the HSMN (Hierarchical State-Space Model Network) Architecture, providing enterprise-grade language model capabilities in an accessible, open-source package. This edition includes the **full architecture** with enforced scale limits—perfect for research, experimentation, and production workloads within the defined bounds.

## 🌟 Key Features

- **Full HSMN Architecture**: Complete model architecture, not a stripped-down version
- **Controlled Scale**: Enforced limits on parameters, experts, and context length
- **Mixture of Experts**: Up to 12 experts with Expert Choice routing
- **Quantum-Inspired Components**: Wavelet tokenization and graph learning
- **Agentic Capabilities**: Built-in Codex CLI for tool-using agents
- **WebUI**: Visual curriculum builder, training console, and HPO dashboard

## 📈 Benchmark Results

> [!NOTE]
> **Benchmarking in Progress:** Full benchmarks are currently running and showing **very promising results**. Stay tuned for complete reports!

Early benchmark results demonstrate the power of the HSMN architecture:

| Metric | Result | Notes |
|--------|--------|-------|
| **Streaming Inference** | 174-181 tok/s | O(1) memory, constant across context lengths |
| **Batch=2 Inference** | 342-345 tok/s | Near-linear scaling |
| **Context Scaling** | ✅ Constant | 128 → 131K tokens with **no throughput degradation** |
| **Memory Overhead** | ~3.8 GB | For 1M token context |
| **SIMD Optimized** | ✅ Yes | AVX2/FMA acceleration |

**Test Hardware**: AMD Ryzen 7 2700X (8 cores/16 threads, 64GB RAM, no GPU)

The architecture achieves **constant throughput regardless of context length** thanks to:
- **StreamingInferenceWrapper**: O(1) memory forward passes
- **Quantum Superposition Generation (QSG)**: Parallel token generation
- **Fused C++ Kernels**: Optimized native operations

## 📦 Installation

```bash
pip install highnoon (coming soon)
```

Or from source:

```bash
git clone https://github.com/versoindustries/highnoon.git
cd highnoon
pip install -e .
```

### Build Dependencies

For building native C++ operations (optional but recommended for performance):

```bash
# Ubuntu/Debian
sudo apt-get install -y libspdlog-dev cmake build-essential

# macOS
brew install spdlog cmake

# Build native operations
cd highnoon/_native
./build_secure.sh
```

## 🚀 Quick Start

```python
import highnoon

# Create a model from preset
model = highnoon.create_model("highnoon-base")

# Or with custom configuration
from highnoon.config import Config, ModelConfig

config = Config(
    model=ModelConfig(
        vocab_size=32000,
        embedding_dim=768,
        num_reasoning_blocks=8,
        num_moe_experts=8,
    )
)
model = highnoon.create_model(config)

# Training
from highnoon.training import Trainer

trainer = Trainer(model, learning_rate=1e-4)
trainer.train(train_dataset, epochs=10)

# Generation
output = model.generate("Hello, world!", max_length=100)
print(output)
```

## 🏗️ Architecture Overview

```
HighNoon Framework
├── Models
│   ├── HSMN (Full architecture)
│   ├── HSMNLanguageModel (Lightweight variant)
│   └── MoELayer (Mixture of Experts)
├── Reasoning
│   ├── ReasoningModule (Multi-layer reasoning)
│   └── MemoryBuilder (Hierarchical memory)
├── Training
│   ├── Trainer (Training orchestrator)
│   └── CurriculumScheduler (Curriculum learning)
├── CLI
│   ├── CodexRunner (Agent loop)
│   └── ToolManifest (Tool registry)
└── Services
    ├── ACI (Agent Communication Interface)
    └── Tooling (Simulation dispatch)
```

## 📊 Lite Edition Limits

> [!CAUTION]
> **Enforced Limits:** The Lite Edition includes tamper-proof compiled binaries that enforce the following hard limits. These cannot be modified or bypassed. For unlimited scale, contact us about the Enterprise Edition.

| Feature | Lite (This Package) | Enterprise |
|---------|---------------------|------------|
| Architecture | Full HSMN | Full HSMN |
| Max Parameters | **20B** | Unlimited |
| Reasoning Blocks | **24** | Unlimited |
| MoE Experts | **12** | Unlimited |
| Context Length | **5M tokens** | Unlimited |
| Binary Modification | ❌ Disabled | ✅ Configurable |
| Domain Modules | General | All Domains |
| Support | Community | Dedicated |

## 🔧 Configuration

HighNoon uses a hierarchical configuration system:

```python
from highnoon.config import Config, ModelConfig, TrainingConfig

# Model configuration
model_config = ModelConfig(
    vocab_size=32000,
    embedding_dim=768,
    num_reasoning_blocks=8,
    max_seq_length=4096,
    num_moe_experts=8,
)

# Training configuration
training_config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=8,
    epochs=100,
    warmup_steps=1000,
)

# Combined configuration
config = Config(model=model_config, training=training_config)
```

## 🤖 Agentic Capabilities

HighNoon includes the Codex CLI for agent-based workflows:

```python
from highnoon.cli import CodexRunner, ToolManifest

# Create manifest with default tools
manifest = ToolManifest()

# Create runner
runner = CodexRunner(model=model, manifest=manifest)

# Run agent loop
response = runner.run("Analyze the code in src/models/")
print(response)
```

## 🌐 Distributed Training

Train across multiple CPU nodes with built-in distributed training support:

```python
from highnoon.training.distributed import create_cpu_strategy

# Auto-detect cluster from TF_CONFIG
strategy = create_cpu_strategy()

with strategy.scope():
    model = highnoon.create_model("7b")
    trainer = Trainer(model, learning_rate=1e-4)
    trainer.train(checkpoint_dir="/shared/checkpoints")
```

**Features:**
- Multi-node synchronous training (MultiWorkerMirroredStrategy)
- Automatic dataset sharding
- Fault-tolerant checkpointing
- CPU-optimized threading configuration

**Guides:**
- [Distributed Training Guide](docs/distributed_training.md) - Full setup instructions
- [Cluster Setup](docs/cluster_setup.md) - SLURM, Kubernetes, bare-metal

## 🖥️ WebUI

HighNoon includes a full-featured WebUI for training management:

```bash
# Start the WebUI
python -m highnoon.webui --host 0.0.0.0 --port 8000
```

**Features:**
- **Curriculum Builder** - Visual drag-and-drop stage management
- **Training Console** - Real-time metrics, loss charts, and controls
- **HPO Dashboard** - Hyperparameter optimization with Optuna integration
- **Model Browser** - View, export, and manage trained models
- **REST API** - Programmatic access to all features

See the [WebUI Guide](docs/guides/webui.md) for full documentation.

## 📚 Documentation

- **API Reference**: Coming soon
- **Tutorials**: Coming soon
- **Examples**: See the `examples/` directory

## 🏢 Enterprise Edition

For production deployments requiring:
- Unlimited scale
- Domain-specific modules (Finance, Healthcare, Legal, Scientific)
- Dedicated support
- Custom integrations

Contact: [www.versoindustries.com/messages](https://www.versoindustries.com/messages)

## 📄 License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

HighNoon is developed by [Verso Industries](https://versoindustries.com) and builds upon research in:
- State-space models (Mamba, S4)
- Mixture of Experts architectures
- Quantum-inspired machine learning
- Hierarchical memory systems

## 🤝 Contributing

We welcome contributions from the community! This project is under **active development** and we're excited to collaborate.

**Ways to Contribute:**
- 🐛 Report bugs and issues
- 💡 Suggest features and improvements
- 📖 Improve documentation
- 🧪 Add tests and benchmarks
- 🔧 Submit pull requests

See our [Contributing Guide](CONTRIBUTING.md) for details on the development workflow.

> [!TIP]
> **Star this repository** ⭐ to show your support and stay updated on new releases!

## 📞 Support

- **Community Support**: [www.versoindustries.com/messages](https://www.versoindustries.com/messages)
- **Enterprise Support**: Available with Pro or Enterprise editions

---

<p align="center">
  <strong>Built with ❤️ by Verso Industries</strong>
</p>
