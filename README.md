<div align="center">

# Lumyn: Solana-Native Any-to-Any AI Platform
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---
### Bridging Multimodal AI with Blockchain Performance
---
</div>

- [Introduction](#introduction)
- [Architecture](#architecture)
- [Why Lumyn?](#why-Lumyn)
- [Roadmap](#roadmap)
- [Getting Started](#getting-started)
  - [For Validators](#for-validators)
  - [For Developers](#for-developers)
- [Technical Design](#technical-design)
- [Future Vision](#future-vision)

---

## Introduction

**  ** is a groundbreaking decentralized AI infrastructure built natively on Solana, designed to create state-of-the-art (SOTA) multimodal any-to-any models. By leveraging Solana's high-performance blockchain and advanced smart contract capabilities, Lumyn establishes a new paradigm for decentralized AI computation and model training.

## Architecture

Lumyn's architecture is built on three core pillars:

### 1. Solana Programs (Smart Contracts)
- **Model Registry**: On-chain program managing model versioning and access control
- **Computation Market**: Automated matching of compute providers with training tasks
- **Reward Distribution**: Performance-based incentive distribution system
- **Governance**: DAO-controlled protocol upgrades and parameter adjustment

### 2. Any-to-Any AI Pipeline
- Native multimodal processing (text, image, audio, video)
- Distributed model training across validator network
- Cross-modal translation and understanding
- Composable AI programs for specialized tasks

### 3. Network Infrastructure
- Decentralized storage integration with Arweave
- High-throughput validator nodes
- Real-time model serving infrastructure
- Cross-chain bridges for expanded reach

## Why Lumyn? 🧠

- **Solana-Native Speed**: Sub-second finality and minimal fees enable real-time AI model updates and micro-payments for compute
- **True Decentralization**: Leveraging Solana's proof-of-history for deterministic model validation
- **Multimodal First**: Joint modeling of all modalities through Solana's parallel processing capabilities
- **Incentivized Research**: Smart contract-based reward system for researchers and compute providers
- **Scalable Infrastructure**: Solana's 65,000 TPS enables enterprise-grade AI serving
- **Composable AI**: Build complex AI applications using Solana's composable programming model

## Roadmap 🚀

### Phase 1: Foundation (Q4 2024)
- [x] Launch core Solana programs for model registry and compute market
- [x] Integrate with major Solana DeFi protocols for liquidity provision
- [x] Deploy advanced architecture search capabilities
- [ ] Establish validator network with 100+ nodes
- [ ] Implement Solana-native reward distribution system
- [ ] Release comprehensive documentation and miner guides

### Phase 2: Expansion (Q1 2025)
- [x] Deploy initial multimodal model architecture with Llama-3 backbone
- [ ] Implement cross-chain asset bridging via wormhole
- [ ] Expand to 1000+ validator nodes
- [ ] Release the Lumyn SDK for enterprise AI integration

### Phase 3: Ecosystem (Q2 2025)
- [ ] Deploy quantum-resistant cryptographic layer
- [ ] Launch Lumyn App Store for AI applications
- [ ] Implement advanced tokenomics model with recursive rewards
- [ ] Enable composable AI programs with zero-knowledge proofs
- [ ] Achieve mainnet stability with 10,000+ TPS
- [ ] Launch decentralized model marketplace

## Getting Started 🏁

### For Miners

#### Requirements
- Python 3.11+ with pip
- GPU with at least 40 GB of VRAM; NVIDIA RTXA6000 is a good choice, or use a 1024xH100 if you wanna train a **really** good model :sunglasses:
- At least 40 GB of CPU RAM
- If running on runpod, `runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04` is a good base template.

#### Setup
1. Clone the repo and `cd` into it:
```bash
git clone https://github.com/Lumyn-ai/Lumyn
cd Lumyn-ai
```
2. Install the requirements:
  - Using docker: `make build-and-run`
  - Using your local Python: `pip install -e .`
3. Log into Huggingface: `huggingface-cli login`. Make sure your account has access to Llama-3-8B on HF
4. Download the base model and datasets: `make download-everything`
5. Start a finetuning run: `make finetune-x1`
  - Tweak `config/8B_lora.yaml` to change the hyperparameters of the training run.

### For Validators

#### Requirements
- Python 3.11+ with pip
- GPU with at least 40 GB of VRAM; NVIDIA RTXA6000 is a good choice
- At least 40 GB of CPU RAM
- At least 300 GB of free storage space
- If running on runpod, `runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04` is a good base template.
- Install libatlas-base-dev: `apt install libatlas-base-dev`

#### Running with Docker
1. Clone the repo and `cd` into it:
```bash
git clone https://github.com/Lumyn-ai/Lumyn
cd Lumyn-ai
```
2. Run the validator:
```bash
make validator WALLET_NAME={wallet} WALLET_HOTKEY={hotkey} PORT={port}
```
##### Recommended
- Setting up wandb. Open the `vali.env` file in the repo root directory and set the `WANDB_API_KEY`. Alternatively, you can disable W&B with `WANDB=off` in Step 2.

3. Check your logs: `make check-vali-logs`

## Technical Design 🔧

Lumyn's technical architecture combines Solana's parallel processing capabilities with advanced AI infrastructure:

- **Parallel Model Training**: Utilize Solana's proof-of-history for deterministic model updates
- **On-Chain Governance**: Smart contract-based parameter optimization
- **Composable AI Programs**: Build complex AI applications using Solana's program composition
- **Real-Time Serving**: Sub-second model inference using Solana's high throughput

## Future Vision 🔮

Lumyn aims to become the backbone of decentralized AI infrastructure by:
- Enabling enterprise-grade AI applications on Solana
- Creating a thriving marketplace for specialized AI models
- Establishing a self-sustaining research ecosystem
- Powering the next generation of multimodal AI applications

Join us in building the future of decentralized AI on Solana!
