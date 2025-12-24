# HighNoon Language Framework v1.0.0

🎉 **Initial Public Release**

HighNoon is an enterprise-grade language modeling framework providing
the complete HSMN architecture with controlled scale limits.

---

## Highlights

- 🧠 **Full HSMN Architecture**: Complete MoE, Quantum, and Reasoning modules
- 📚 **Curriculum Learning**: Multi-stage training with configurable schedules
- 🤖 **Codex CLI Agent**: Built-in tool-using agent with manifest system
- 🎛️ **WebUI Dashboard**: Curriculum builder and training console
- ⚡ **Native C++ Operations**: Performance-optimized operators
- 🔒 **Binary Security**: Tamper-proof compiled binaries

---

## Scale Limits (Lite Edition)

| Limit | Value |
|-------|-------|
| Max Parameters | 20B |
| Max Reasoning Blocks | 24 |
| Max MoE Experts | 12 |
| Max Context Length | 5M tokens |

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

---

## Quick Start

```python
import highnoon as hn

# Create a model
model = hn.create_model("highnoon-3b")

# Generate text
response = model.generate("Hello, world!")
print(response)

# Use the Codex CLI agent
runner = hn.CodexRunner(model)
result = runner.run("What is the capital of France?")
print(result)
```

---

## What's Included

### Models
- `HSMN` - Full hierarchical state-space model network
- `HSMNLanguageModel` - Lightweight language-focused variant
- `MoELayer` - Mixture of Experts layer with Expert Choice routing

### Training
- `Trainer` - Training orchestrator with curriculum support
- `CurriculumScheduler` - Multi-stage curriculum learning
- `HPOTrialManager` - Hyperparameter optimization

### CLI
- `CodexRunner` - Agent execution loop
- `ToolManifest` - Tool registration and management

### WebUI
- Curriculum builder interface
- Training console with real-time metrics
- HPO dashboard

---

## Requirements

- Python 3.10+
- TensorFlow 2.15+
- numpy >= 1.24.0

### Optional Dependencies

```bash
# For development
pip install highnoon[dev]

# For training (includes tensorboard, transformers, datasets)
pip install highnoon[training]

# Everything
pip install highnoon[full]
```

---

## Documentation

- [Documentation Index](index.md)
- [Getting Started](getting-started.md)
- [API Reference](api/)
- [User Guides](guides/)
- [Enterprise Upgrade Guide](enterprise-upgrade.md)
- [Examples](../examples/)

---

## Enterprise Edition

Need unlimited scale? The Enterprise edition removes all limits and includes:

- Unlimited parameters, context length, and experts
- Chemistry, Physics, and Inverse Design modules
- Full source code access
- 24/7 dedicated support

[Contact us for enterprise licensing](https://versoindustries.com/enterprise)

---

## License

- **Python Layer**: Apache 2.0 with attribution requirements
- **Compiled Binaries**: Proprietary (see LICENSE for details)
- **Documentation**: CC BY-SA 4.0

---

## Links

- **Website**: [versoindustries.com](https://versoindustries.com)
- **Support**: [versoindustries.com/messages](https://versoindustries.com/messages)
- **Enterprise Sales**: sales@versoindustries.com

---

*Powered by HSMN • Built by Verso Industries*
