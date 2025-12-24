# Hardware Specification Report

**Generated**: 2025-12-24T01:40:30.106041

---

## CPU

| Specification | Value |
|---------------|-------|
| **Model** | AMD Ryzen 7 2700X Eight-Core Processor |
| **Vendor** | AMD |
| **Architecture** | x86_64 |
| **Physical Cores** | 8 |
| **Logical Cores** | 16 |
| **Base Frequency** | 3486 MHz |
| **Max Frequency** | 3700 MHz |
| **L1d Cache** | 256 KB/core |
| **L2 Cache** | 4 KB/core |
| **L3 Cache** | 16 KB |
| **SIMD Extensions** | AVX2, AVX, SSE4.2, SSE4.1, FMA, F16C |
| **Governor** | schedutil |
| **NUMA Nodes** | 1 |

## Memory

| Specification | Value |
|---------------|-------|
| **Total RAM** | 62.7 GB |
| **Available** | 51.2 GB |
| **Used** | 11.5 GB |
| **Type** | Unknown |
| **Speed** | 0 MT/s |
| **Swap Total** | 8.0 GB |

## GPU

| Specification | Value |
|---------------|-------|
| **Model** | No GPU detected |
| **Vendor** | None |
| **VRAM** | 0.0 GB |
| **Compute Capability** | N/A |
| **Driver** | N/A |
| **CUDA** | N/A |

## Storage

| Specification | Value |
|---------------|-------|
| **Type** | NVMe SSD |
| **Total** | 915 GB |
| **Free** | 653 GB |
| **Filesystem** | ext4 |

## Operating System

| Specification | Value |
|---------------|-------|
| **OS** | Linux |
| **Distribution** | Ubuntu 24.04.3 LTS |
| **Kernel** | 6.14.0-37-generic |
| **Python** | 3.12.3 |
| **TensorFlow** | Skipped (fast mode) |

## Benchmark Suitability

| Metric | Value |
|--------|-------|
| **Max Recommended Context** | 1,000,000 tokens |
| **Max Recommended Batch Size** | 8 |
| **Can Run 1M Context** | ✅ Yes |
| **Est. 1M Context Memory** | 3.8 GB |
| **SIMD Optimized** | ✅ Yes |
