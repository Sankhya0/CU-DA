# Continual Unsupervised Domain Adaptation (CUDA) - Approach 3: LoRA-DAD with FixMatch & SHOT

**Authors:** Divye Dixit, Samvaidan Salgotra

A comprehensive PyTorch implementation of **Approach 3** from the Master's thesis project on *Continual Unsupervised Domain Adaptation*, completed at IIT Mandi.

---

## 📋 Overview

This repository contains a state-of-the-art implementation for handling **domain shift** in scenarios where:
- You have a **labeled source domain** for initial training
- You need to adapt to **multiple unlabeled target domains sequentially**
- You want to minimize **catastrophic forgetting** on previously adapted domains
- You need to maintain **low computational overhead**

### The Problem: Continual Unsupervised Domain Adaptation

Traditional machine learning models suffer from **domain shift** when deployed in environments different from their training data. This project addresses a particularly challenging setting: **Continual Unsupervised Domain Adaptation (CUDA)**, where:

1. A labeled source domain is available initially
2. Multiple unlabeled target domains appear sequentially with different distributions
3. The model must adapt to each new domain without labels
4. Previous domain performance should not degrade (avoid catastrophic forgetting)

---

## 🎯 Approach 3: LoRA-DAD with FixMatch & SHOT

This implementation combines cutting-edge techniques to achieve **state-of-the-art results on a low compute budget**:

### Key Components

#### 1. **Vision Transformer (ViT) Backbone**
- Uses pretrained `vit_base_patch16_224` from timm library
- Frozen base model to preserve learned representations
- Efficient feature extraction with minimal parameters

#### 2. **Low-Rank Adaptation (LoRA)**
- Injects trainable low-rank decomposition into ViT attention layers
- Significantly reduces trainable parameters
- Enables fast adaptation across domains
- **Configuration:**
  - Rank: 16
  - Alpha: 32.0
  - Dropout: 0.05

#### 3. **Domain-Adaptive Diffusion (DAD)**
- Aligns source and target feature distributions
- Uses diffusion-based feature reconstruction
- Mitigates catastrophic forgetting through feature alignment
- **Configuration:**
  - Diffusion steps: 200
  - Beta schedule: Linear from 1e-4 to 2e-2

#### 4. **EMA Teacher Framework**
- Maintains exponential moving average (EMA) of student weights
- Provides pseudo-labels for unlabeled target data
- Decay rate: 0.999
- Adaptive threshold: 0.7 → 0.9

#### 5. **FixMatch Semi-Supervised Learning**
- Combines weak and strong augmentations
- Enforces consistency between weak and strong augmented samples
- Confidence threshold: 0.95
- Enables learning from unlabeled target data

#### 6. **SHOT (SHallow Target Optimization)**
- Entropy minimization without source data (optional, can be source-free)
- Reduces decision boundary uncertainty
- Two terms:
  - Conditional entropy minimization: λ = 0.05
  - Marginal entropy maximization: λ = 0.05

#### 7. **Domain Classifier**
- Trained to identify domain identity
- Used in robust inference pipeline for expert selection
- Architecture: ViT backbone → Linear classifier (num_domains outputs)

---

## 🏗️ Architecture Overview

### Training Pipeline

```
Source Domain (Art)
    ↓
[Train LoRA-adapted ViT + Head on labeled source]
    ↓
Target Domain 1 (Clipart)
    ↓
[Multi-Stage Adaptation]:
  - DAD LTR Pre-training (Diffusion alignment)
  - Main Adaptation Loop (200 DAD steps):
    - MLS Diffusion Consistency (p_θ updates)
    - EMA Teacher Pseudo-labeling
    - FixMatch Semi-supervised loss
    - SHOT Entropy minimization
    ↓
Target Domain 2 (Product) → Repeat with warm-start from Domain 1
    ↓
Target Domain 3 (RealWorld) → Repeat with warm-start from Domain 2
```

### Inference Pipeline (3-Stage Robust)

```
Input Image
    ↓
Stage 1: Domain Classification
  - Identify target domain using Domain Classifier
  - If high confidence → use corresponding expert directly
  ↓
Stage 2: Lite Test-Time Augmentation (TTA-Lite)
  - 2 augmentations (original + flipped)
  - Average expert predictions
  - If high confidence → return prediction
  ↓
Stage 3: Full Test-Time Augmentation (TTA-Full)
  - 5 augmentations (more aggressive)
  - Average ensemble of top-k experts
  - Final prediction
```

---

## 📊 Key Features

### Performance
- **State-of-the-art accuracy** on OfficeHome benchmark
- **Low computational cost** with LoRA (only ~10% of full ViT parameters)
- **Efficient inference** with multi-stage confidence-based routing
- **Minimal forgetting** through domain-adaptive diffusion

### Robustness
- **Multi-stage inference** with confidence thresholds
- **Test-Time Augmentation (TTA)** for improved generalization
- **Expert ensemble** for uncertain samples
- **Domain classification** for intelligent expert routing

### Flexibility
- **Modular design** for easy extension
- **Configurable hyperparameters** for different datasets
- **Optional replay buffer** for experience replay
- **Warm-start transfer** from previous domain experts

---

## 📈 Visual Results & Comparisons

### Approach Comparison
**All three approaches analyzed, with Approach 3 achieving state-of-the-art results:**

![Approach Comparison](Artifacts/1.%20comparison_list.png)

*Figure 1: Side-by-side comparison of Approaches 1, 2, and 3 across all metrics*

### Accuracy Progression Across Domains
**Demonstrates strong source training and effective target adaptation with minimal forgetting:**

![Accuracy Progression](Artifacts/2.%20graph_1.png)

*Figure 2: Accuracy progression as model adapts to each domain sequentially (Art → Clipart → Product → RealWorld)*

### Forgetting vs Adaptation Trade-off

![Forgetting Trade-off](Artifacts/3.%20graph_2.png)

*Figure 3: Forgetting vs adaptation strength analysis showing Approach 3's superior trade-off curve*

---

## 🚀 Quick Start

### Requirements
```bash
pip install torch torchvision timm einops
```

### Basic Usage

The notebook is designed to run end-to-end. Here's the typical workflow:

#### 1. **Configure Hyperparameters** (Cell 3)
```python
# Dataset
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 65

# LoRA
LORA_RANK = 16
LORA_ALPHA = 32.0

# DAD
DAD_K_STEPS = 200

# Training
ART_EPOCHS = 10  # Source domain
ADAPT_LTR_EPOCHS = 5  # DAD pre-training
```

#### 2. **Train Source Expert** (Cells 13-15)
```python
# Loads OfficeHome Art domain
# Trains LoRA-adapted ViT + classification head
# Saves best model based on validation accuracy
```

#### 3. **Continual Adaptation** (Cell 17)
```python
# For each target domain (Clipart, Product, RealWorld):
#   - DAD LTR pre-training
#   - Multi-stage adaptation with 200 DAD steps
#   - Early stopping based on validation accuracy
#   - Warm-start from previous domain
```

#### 4. **Domain Classifier Training** (Cell X2)
```python
# Trains to classify which domain an image belongs to
# Used in robust inference pipeline
```

#### 5. **Robust Inference Evaluation** (Cell X5)
```python
# Evaluates 3-stage inference pipeline
# Reports domain-wise and overall accuracy
```

---

## 📁 Notebook Structure

| Cell | Purpose |
|------|---------|
| 1-2 | Imports and utility setup |
| 3 | Configuration and hyperparameters |
| 4 | Data transforms (strong/weak augmentations) |
| 5-6 | Dataset classes and global class mapping |
| 7 | ViT backbone loading |
| 8 | LoRA injection and implementation |
| 9 | Domain-specific heads |
| 10-11 | DAD (Diffusion) components |
| 12 | EMA updates and replay buffer |
| 13-15 | **Source domain (Art) training** |
| 15b | Baseline validation |
| 16 | Continual learning preparation |
| 17 | **Main continual adaptation loop** |
| X1-X2 | **Domain classifier training** |
| X3-X5 | **Robust inference pipeline** |

---

## 📈 Results

### Performance Metrics

The implementation achieves:
- **Strong source domain accuracy** (~95%+ on Art)
- **Effective target domain adaptation** with 5-15% improvement over baseline
- **Minimal forgetting** on previously adapted domains
- **Robust inference** with multi-stage ensemble

Example results structure:
```
Domain-wise Accuracies:
  Art (Source): 95.5%
  Clipart (Target 1): 82.3%
  Product (Target 2): 88.7%
  RealWorld (Target 3): 91.2%

Overall Accuracy: 89.4%
```

---

## 🎨 Artifacts & Visualizations

The `Artifacts/` folder contains:

1. **comparison_list.png** - Side-by-side comparison of different approaches
2. **graph_1.png** - Accuracy progression across domains
3. **graph_2.png** - Forgetting vs adaptation trade-off analysis

---

## 🔬 Technical Details

### Loss Functions

The main adaptation combines several objectives:

```
Total Loss = L_MLS + L_PL + L_FixMatch + L_SHOT

where:
  L_MLS         = Diffusion consistency loss
  L_PL          = EMA teacher pseudo-label loss
  L_FixMatch    = Semi-supervised consistency loss
  L_SHOT        = Entropy minimization loss
```

### DAD Diffusion Process

```
Forward (noising):
  x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε

Reverse (denoising):
  x̂_0 = (x_t - √(1 - ᾱ_t) * ε̂_θ(x_t, t)) / √(ᾱ_t)

Training:
  L = MSE(ε, ε̂_θ(x_t, t))
```

### Inference Confidence Thresholds

```
Stage 1 (Domain Classifier): 0.7
Stage 2 (Expert Confidence): 0.6
Stage 3 (Expert Confidence): 0.5
TTA Lite: 2 augmentations
TTA Full: 5 augmentations
K experts for ensemble: 3
```

---

## 💾 Model Saving & Loading

Models are automatically saved to:
```
models/
├── Art/                          # Source domain
│   ├── art_vit_lora_best.pth
│   └── art_head_best.pth
├── Clipart/                      # Target domains
│   ├── clipart_vit_lora_best.pth
│   └── clipart_head_best.pth
├── Product/
│   ├── product_vit_lora_best.pth
│   └── product_head_best.pth
├── RealWorld/
│   ├── realworld_vit_lora_best.pth
│   └── realworld_head_best.pth
└── domain_classifier_head_best.pth
```

---

## 🔧 Customization

### For Different Datasets

1. Modify `DATA_DIR` to point to your dataset
2. Update `NUM_CLASSES` to match your task
3. Adjust `IMAGE_SIZE` if needed
4. Modify domain names in `TARGET_DOMAIN_NAMES_ORDERED`

### For Different Domains

```python
SOURCE_DOMAIN_NAME = 'YourSourceDomain'
TARGET_DOMAIN_NAMES_ORDERED = ['Target1', 'Target2', 'Target3']
```

### Hyperparameter Tuning

Key parameters for performance:
- `LORA_RANK`: Lower = faster, Higher = more expressive
- `DAD_K_STEPS`: More steps = better alignment, slower training
- `EMA_DECAY`: Higher = more stable, slower updates
- `FIXMATCH_CONF_THRESHOLD`: Higher = more selective, fewer samples

---

## 📚 References

### Paper & Thesis
- **Title:** Continual Unsupervised Domain Adaptation
- **Authors:** Divye Dixit, Samvaidan Salgotra
- **Institution:** IIT Mandi, School of Computing and Electrical Engineering (SCEE)
- **Submitted:** May 2025

### Key Literature Cited
1. [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
2. [FixMatch: Simplifying Semi-Supervised Learning](https://arxiv.org/abs/2001.04451)
3. [SHOT: Do We Really Need to Access the Source Data?](https://arxiv.org/abs/2002.08546)
4. [Domain-Adaptive Diffusion for Unsupervised Domain Adaptation](https://arxiv.org/abs/2308.13893)
5. [Exponential Moving Average Teacher (Mean Teacher)](https://arxiv.org/abs/1703.01026)

---

## 🎓 Approach Contributions

### What Makes Approach 3 Special

**Compared to traditional UDA methods:**
- ✅ Handles **multiple sequential domains** (not just source→target)
- ✅ **Prevents catastrophic forgetting** through diffusion-based alignment
- ✅ **Extremely parameter-efficient** with LoRA (only ~10% parameters)
- ✅ **Low computational cost** suitable for resource-constrained settings
- ✅ **No source data needed** during adaptation (can be source-free)

**Compared to Approaches 1 & 2:**
- **Approach 1:** Feature-level diffusion replay - explicit replay required
- **Approach 2:** LoRA expert modules - separate experts per domain, less adaptation
- **Approach 3:** Integrated solution with minimal parameters, better accuracy

---

## 🤝 Citation

If you use this code in your research, please cite:

```bibtex
@thesis{Dixit2025CUDA,
  title={Continual Unsupervised Domain Adaptation},
  author={Dixit, Divye and Salgotra, Samvaidan},
  year={2025},
  school={Indian Institute of Technology Mandi},
  note={Master's Thesis, School of Computing and Electrical Engineering}
}
```

---

## 📝 Implementation Notes

- **PyTorch Version:** Compatible with PyTorch 1.9+
- **CUDA:** Optimized for CUDA but works on CPU (slower)
- **Mixed Precision:** Uses AMP (Automatic Mixed Precision) for efficiency
- **Data Parallel:** Can be extended to multi-GPU training

---