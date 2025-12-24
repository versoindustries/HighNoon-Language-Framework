# HighNoon Language Framework Documentation

> **Language Intelligence, Engineered** | Powered by HSMN

Welcome to the HighNoon Language Framework documentation. HighNoon is the **Lite Edition** of the HSMN (Hierarchical State-Space Model Network) Architecture, providing enterprise-grade language model capabilities.

---

## Quick Links

| Getting Started | Core Concepts | Reference |
|-----------------|---------------|-----------|
| [Installation](getting-started.md#installation) | [Models](api/models.md) | [Configuration](api/config.md) |
| [Quick Start](getting-started.md#quick-start) | [Training](guides/training.md) | [API Reference](api/) |
| [First Model](getting-started.md#your-first-model) | [Curriculum Learning](guides/curriculum-learning.md) | [Examples](../examples/) |

---

## What is HighNoon?

HighNoon provides the **complete HSMN architecture** with controlled scale limits enforced through compiled binaries:

- 🧠 **Full Architecture**: Complete MoE, Quantum-inspired, and Reasoning modules
- 📚 **Curriculum Learning**: Multi-stage training with configurable schedules
- 🤖 **Codex CLI Agent**: Built-in tool-using agent with manifest system
- 🎛️ **WebUI Dashboard**: Curriculum builder and training console
- ⚡ **Native Operations**: Performance-optimized compiled operators

---

## Edition Comparison

| Feature | Lite (This Package) | Enterprise |
|---------|---------------------|------------|
| **Architecture** | Full HSMN | Full HSMN |
| **Max Parameters** | 20B | Unlimited |
| **Max Reasoning Blocks** | 24 | Unlimited |
| **Max MoE Experts** | 12 | Unlimited |
| **Max Context Length** | 5M tokens | Unlimited |
| **Domain Modules** | Language only | Chemistry, Physics, Inverse Design |
| **Support** | Community | 24/7 SLA |

[Learn about Enterprise →](enterprise-upgrade.md)

---

## Documentation Sections

### Getting Started
- [Installation & Setup](getting-started.md)
- [Quick Start Guide](getting-started.md#quick-start)
- [Your First Model](getting-started.md#your-first-model)

### API Reference
Complete reference for all public APIs:
- [Models](api/models.md) - `create_model()`, `HSMN`, presets
- [Training](api/training.md) - `Trainer`, `CurriculumScheduler`
- [Inference](api/inference.md) - Generation, streaming, QSG
- [CLI](api/cli.md) - `CodexRunner`, `ToolManifest`
- [Configuration](api/config.md) - `Config`, `ModelConfig`, `TrainingConfig`
- [Tokenization](api/tokenization.md) - `QWTTextTokenizer`

### User Guides
Step-by-step guides for common workflows:
- [Training Workflow](guides/training.md)
- [Curriculum Learning](guides/curriculum-learning.md)
- [Agent Tools](guides/agent-tools.md)
- [Hyperparameter Optimization](guides/hpo.md)
- [WebUI Dashboard](guides/webui.md)

### Advanced Topics
- [Distributed Training](distributed_training.md)
- [Cluster Setup](cluster_setup.md)

---

## Installation

```bash
pip install highnoon
```

Or from source:

```bash
git clone https://github.com/versoindustries/HighNoon-Language-Framework.git
cd HighNoon-Language-Framework
pip install -e .
```

[Full installation guide →](getting-started.md#installation)

---

## Quick Example

```python
import highnoon as hn

# Create a model
model = hn.create_model("highnoon-3b")

# Generate text
response = model.generate("Explain quantum computing", max_length=256)
print(response)

# Train with curriculum learning
trainer = hn.Trainer(model, learning_rate=1e-4)
trainer.add_curriculum_stage("foundation", datasets=["your_dataset"])
trainer.train(epochs_per_stage=5)
```

---

## Support

- **Community**: [versoindustries.com/messages](https://versoindustries.com/messages)
- **Enterprise**: [sales@versoindustries.com](mailto:sales@versoindustries.com)
- **Documentation Issues**: GitHub Issues

---

*Built with ❤️ by Verso Industries*
