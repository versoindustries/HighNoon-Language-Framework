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
> **CPU-Native Architecture:** HighNoon is purpose-built for **pure CPU execution**. No GPU required. Run production AI on commodity hardware—from data centers to edge deployments.

> [!NOTE]
> **Binary Distribution:** The Lite Edition includes pre-compiled, hardened binaries. Users run the setup script and immediately start building models through the WebUI—no compilation required.

---

## 🔗 Community & Resources

<p align="center">
  <a href="https://huggingface.co/versoindustries"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Models-yellow.svg" alt="HuggingFace"/></a>
  <a href="https://github.com/versoindustries/HighNoon-Language-Framework"><img src="https://img.shields.io/badge/GitHub-Repository-181717.svg?logo=github" alt="GitHub"/></a>
  <a href="https://versoindustries.com"><img src="https://img.shields.io/badge/Verso-Industries-blue.svg" alt="Verso Industries"/></a>
</p>

- **🤗 HuggingFace**: [huggingface.co/versoindustries](https://huggingface.co/versoindustries) — Pre-trained models and datasets
- **📖 Documentation**: [docs/](docs/) — Architecture guides and API reference
- **💬 Contact**: [versoindustries.com/messages](https://www.versoindustries.com/messages) — Enterprise inquiries and support

---

## 🚀 What is HighNoon?

**HighNoon** is the **Lite Edition** of the HSMN (Hierarchical State-Space Model Network) Architecture—a radical departure from GPU-dependent Transformer models. Built on **Hyperdimensional Computing** and quantum-inspired algorithms, HighNoon delivers:

- **O(L) Linear Complexity** — Process 5 million token contexts without quadratic scaling
- **Zero GPU Dependency** — Full-stack execution on commodity CPUs (Intel Xeon, AMD EPYC)
- **100–200x Lower Energy** — Run inference on 100W servers instead of 700W GPU clusters
- **50–100x Faster Generation** — Quantum Superposition Generation (QSG) parallel decoding

### The CPU-Native Paradigm

Traditional AI demands H100 GPUs at $25,000–40,000 each, megawatt data centers, and 12+ month procurement timelines. HighNoon runs on hardware you already own:

| Aspect | Traditional (GPU) | HighNoon (CPU) |
|--------|-------------------|----------------|
| **Hardware Cost** | $350,000/node (8×H100) | $30,000/node (Dual EPYC) |
| **Power Draw** | 10kW per node | ~500W per node |
| **Procurement Time** | 12–52 weeks | Available now |
| **Export Restrictions** | Subject to ITAR/EAR | Commodity hardware |

---

## ⚡ Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/versoindustries/HighNoon-Language-Framework.git
cd HighNoon-Language-Framework

# Run the setup script (creates venv, installs dependencies)
./scripts/setup.sh
```

### 2. Launch the WebUI

```bash
# Start the React Dashboard + FastAPI Backend
./scripts/launch_webui.sh
```

Access the WebUI at **http://localhost:5173**

### 3. Build Your First Curriculum

The WebUI provides an integrated workflow for training language models:

1. **Select a Template** — Choose from pre-built curriculum presets (General Pre-training, Coding, Math/Reasoning, etc.)
2. **Browse HuggingFace Datasets** — Search and add datasets directly from the HuggingFace Hub
3. **Configure Training** — Set model architecture, hyperparameters, and optimization settings
4. **Launch Training** — Start QAHPO (Quantum Adaptive HPO) to automatically optimize your model

---

## 🖥️ The WebUI: Curriculum Builder

HighNoon's WebUI is the primary interface for building and training models. It features deep **HuggingFace Hub integration** for seamless dataset access.

### Core Features

| Feature | Description |
|---------|-------------|
| **Template Gallery** | Pre-built curriculum presets with curated HuggingFace datasets |
| **Dataset Browser** | Search, preview, and add any HuggingFace dataset |
| **Curriculum Builder** | Drag-and-drop stage management with dataset mixing |
| **QAHPO Dashboard** | Real-time hyperparameter optimization with importance analysis |
| **Cockpit HUD** | Live training metrics, loss charts, and QULS health indicators |
| **Model Export** | Save and export trained models |

### Pre-Built Curriculum Templates

| Template | Focus | Example Datasets |
|----------|-------|------------------|
| General Pre-training | Broad knowledge | FineWeb, Cosmopedia, OpenWebMath |
| Code Expert | Programming | The Stack v2, CodeContests, CodeSearchNet |
| Math & Reasoning | STEM skills | FineMath, GSM8K, NuminaMath |
| Instruction Following | Assistant behavior | OpenAssistant, UltraChat, Orca |
| Sovereign Defense | Air-gapped deployments | Custom classified data |

### HuggingFace Integration

The WebUI connects directly to the HuggingFace Hub API:

- **Search datasets** by name, task type, or license
- **Preview samples** before adding to curriculum
- **Stream training data** directly from HuggingFace servers
- **Automatic format detection** for text, chat, and instruction formats

---

## 🏗️ Architecture: HSMN

The Hierarchical State-Space Model Network achieves linear complexity through four synergistic pillars:

```
                      ┌─────────────────────────────────────────┐
                      │          Reasoning Block × N            │
Input Tokens ──►      │  ┌─────────────────────────────────┐   │
                      │  │  HD Spatial Block (O(L·D log D))│   │
   Holographic        │  │  FFT Bundling • CTQW Spreading  │   │
   Embedding  ──►     │  └─────────────────────────────────┘   │
                      │  ┌─────────────────────────────────┐   │
   Floquet            │  │  HD TimeCrystal Block           │   │ ──► Output
   Position   ──►     │  │  Floquet Dynamics • Symplectic  │   │     Logits
                      │  └─────────────────────────────────┘   │
   Superposition      │  ┌─────────────────────────────────┐   │
   BPE        ──►     │  │  LMWT Attention (O(L log L))    │   │
                      │  │  Multi-Scale Wavelet Transform  │   │
                      │  └─────────────────────────────────┘   │
                      │  ┌─────────────────────────────────┐   │
                      │  │  HD-MoE (O(D) per token)        │   │
                      │  │  Holographic Similarity Routing │   │
                      │  └─────────────────────────────────┘   │
                      └─────────────────────────────────────────┘
```

### Core Components

| Component | Function | Complexity |
|-----------|----------|------------|
| **SpatialHDblock** | Hyperdimensional state-space with FFT bundling | O(L · D log D) |
| **HD TimeCrystal** | Floquet Hamiltonian dynamics for 100+ layer stability | O(L · D) |
| **LMWT Attention** | Learnable Multi-scale Wavelet Transform | O(L log L) |
| **HD-MoE** | Holographic Mixture-of-Experts routing | O(D) per token |
| **QSG Inference** | Quantum Superposition parallel decoding | 50–100x speedup |

### Why It Works

- **Hyperdimensional Embeddings**: Dense holographic vectors replace sparse attention matrices
- **Physics-Aware Training**: Hamiltonian energy conservation prevents gradient instability
- **Quantum Simulation**: Superposition, entanglement, and Born-rule sampling—on classical CPUs

---

## 📊 Lite Edition Specifications

> [!CAUTION]
> **Enforced Limits:** The Lite Edition includes tamper-proof compiled binaries with cryptographic integrity checks. These limits cannot be bypassed.

| Feature | Lite Edition | Enterprise Edition |
|---------|--------------|-------------------|
| **Architecture** | Full HSMN | Full HSMN |
| **Max Parameters** | 20B | Unlimited |
| **Reasoning Blocks** | 24 | Unlimited |
| **MoE Experts** | 12 | Unlimited |
| **Superposition Dimension** | 4 | Unlimited |
| **Context Length** | 5M tokens | Unlimited |
| **Binary Modification** | ❌ Protected | ✅ Configurable |
| **Support** | Community | Dedicated |

---

## 📈 Performance

Benchmarked on AMD Ryzen 7 2700X (8 cores, 64GB RAM, **no GPU**):

| Metric | Result | Notes |
|--------|--------|-------|
| **Streaming Inference** | 174–181 tok/s | O(1) memory across all context lengths |
| **Batch=2 Inference** | 342–345 tok/s | Near-linear scaling |
| **Context Scaling** | ✅ Constant | 128 → 131K → 5M tokens with no degradation |
| **Memory (1M context)** | ~3.8 GB | Fits in commodity server RAM |
| **SIMD Optimization** | ✅ AVX2/FMA | 32-bit optimized for cache efficiency |

---

## 🏢 Enterprise Edition

For production deployments requiring unlimited scale:

- **Unlimited Parameters** — Train and deploy models of any size
- **Unlimited Context** — Beyond 5M tokens for full-document analysis
- **Source Code License** — Full ownership for sovereign deployments
- **Domain Modules** — Finance, Healthcare, Legal, Defense
- **Technology Transfer** — On-premise training with air-gapped security

**Contact:** [versoindustries.com/messages](https://www.versoindustries.com/messages)

---

## 📄 License

Apache License 2.0 — See [LICENSE](LICENSE) for details.

The compiled native binaries are provided as-is under the Lite Edition terms.

---

## 🙏 Acknowledgments

HighNoon is developed by [Verso Industries](https://versoindustries.com) and builds upon research in:

- State-Space Models (Mamba, S4)
- Hyperdimensional Computing
- Quantum-Inspired Machine Learning
- Hamiltonian Neural Networks

---

<p align="center">
  <strong>Built with ❤️ by Verso Industries</strong><br/>
  <em>Sovereign AI for the Post-GPU Era</em>
</p>
