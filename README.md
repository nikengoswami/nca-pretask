# Growing Neural Cellular Automata

Implementation of emoji-to-emoji transformation using Neural Cellular Automata.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [How It Works](#how-it-works)
6. [Training](#training)
7. [Results](#results)
8. [Technical Details](#technical-details)

---

## Overview

This project implements a Growing Neural Cellular Automata (NCA) that learns to transform a lizard emoji 🦎 into a mushroom emoji 🍄 through learned local update rules.

**Key Features:**
- Self-organizing pattern formation
- GPU-accelerated training
- High-resolution outputs (80×80)
- Robust to perturbations
- Minimal parameters (~8K)

---

## Installation

### Prerequisites

- Python 3.10, 3.11, or 3.12
- NVIDIA GPU (optional, for faster training)
- CUDA 11.8 or later (if using GPU)

### Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Install PyTorch with CUDA (for GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install other dependencies
pip install matplotlib imageio tqdm requests pillow numpy
```

**Or use requirements.txt:**

```bash
pip install -r requirements.txt
```

---

## Usage

### Basic Training

```bash
cd scripts
python train.py --emoji mushroom --seed-emoji lizard
```

### Advanced Training

```bash
python train.py \
    --emoji mushroom \
    --seed-emoji lizard \
    --size 80 \
    --steps 20000 \
    --save_every 2000 \
    --viz_every 2000
```

### Visualization

```bash
python visualize.py \
    --checkpoint ../checkpoints/model_final.pt \
    --emoji mushroom \
    --seed-emoji lizard
```

---

## Project Structure

```
nca_pretask/
├── src/                    # Source code
│   ├── models/
│   │   ├── __init__.py
│   │   └── nca.py         # Core NCA model
│   └── utils/
│       ├── __init__.py
│       └── helpers.py     # Helper functions
│
├── scripts/                # Training & visualization
│   ├── train.py           # Main training script
│   └── visualize.py       # Visualization script
│
├── outputs/                # Training outputs
│   ├── seed.png
│   ├── target.png
│   ├── progress_step_*.png
│   └── loss_curve.png
│
├── checkpoints/            # Model checkpoints
│   ├── model_step_*.pt
│   └── model_final.pt
│
├── docs/                   # Documentation
│   ├── README.md          # This file
│   └── QUICKSTART.md      # Quick start guide
│
└── requirements.txt        # Python dependencies
```

---

## How It Works

### 1. Cell State

Each cell has 16 channels:
- **Channels 0-3:** RGBA color (visible)
- **Channels 4-15:** Hidden features (learned)

### 2. Perception

Cells perceive their neighborhood using **Sobel filters**:
- Horizontal gradient detection
- Vertical gradient detection
- Identity (current state)

Output: 48 channels (16 × 3 perceptions)

### 3. Update Network

2-layer neural network:
```
48 channels → 128 hidden → 16 channels
```

Total parameters: **~8,320**

### 4. Stochastic Updates

- Only 50% of cells update each step (fire_rate=0.5)
- Creates organic, life-like growth
- Prevents synchronization artifacts

### 5. Living Mask

- Cells with alpha > 0.1 are "alive"
- Only alive cells and neighbors can update
- Prevents ghost cells

---

## Training

### Training Loop

1. Sample from pool or use seed
2. Run 64-96 random CA steps
3. Compute loss vs target
4. Backpropagate with gradient normalization
5. Update weights
6. Add results to pool

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning rate | 0.002 | Adam optimizer |
| Batch size | 8 | Training batch |
| Image size | 80×80 | High resolution |
| Training steps | 20,000 | Total iterations |
| CA steps | 64-96 | Random per batch |
| Hidden size | 128 | Update network |
| Channels | 16 | Cell state |

### Key Techniques

**Gradient Normalization:**
```python
for p in model.parameters():
    if p.grad is not None:
        p.grad = p.grad / (p.grad.norm() + 1e-8)
```

Essential for stability when backpropagating through many CA steps.

**Sample Pool:**
- Maintains 8 intermediate states
- Prevents overfitting to single trajectory
- Encourages diverse learning

---

## Results

### Training Performance

- **Initial Loss:** ~0.277
- **Final Loss:** ~0.010-0.020
- **Loss Reduction:** ~96%
- **Training Time:** ~1.5 hours (GPU) / 6-8 hours (CPU)

### Visual Quality

The transformation progresses through stages:
1. **Steps 0-50:** Color shift (green → pink/red)
2. **Steps 60-120:** Shape morphing (lizard → mushroom)
3. **Steps 130-190:** Detail refinement (spots, texture)

### Output Files

- `outputs/progress_step_20000.png` - Full transformation sequence
- `outputs/final_step_20000.png` - Final output
- `outputs/loss_curve.png` - Training loss over time
- `checkpoints/model_final.pt` - Trained model weights

---

## Technical Details

### Architecture

**CAModel Class:**
- Perception: Sobel filter convolutions
- Update: 2-layer CNN (128 hidden)
- Stochastic: Random cell updates
- Living: Alpha-based masking

**Loss Function:**
```python
loss = MSE(output[:, :4], target[:, :4])  # RGBA only
overflow = (x - x.clamp(-2.0, 2.0)).abs().sum() * 0.01
total_loss = loss + overflow
```

### GPU vs CPU

| Device | Speed | 20K Steps Time |
|--------|-------|----------------|
| CPU | ~1-2 it/s | 6-8 hours |
| RTX 4050 | ~4 it/s | 1.5 hours |
| RTX 3090 | ~15-20 it/s | 20-30 min |

### Memory Requirements

- **GPU Memory:** ~2-4 GB
- **CPU Memory:** ~4-8 GB
- **Disk Space:** ~500 MB (outputs + checkpoints)

---

## Command Reference

### Training Options

```bash
python scripts/train.py [OPTIONS]

Options:
  --emoji TEXT              Target emoji name [default: mushroom]
  --seed-emoji TEXT         Seed emoji name [default: lizard]
  --size INTEGER            Image size [default: 80]
  --steps INTEGER           Training steps [default: 20000]
  --batch_size INTEGER      Batch size [default: 8]
  --lr FLOAT               Learning rate [default: 0.002]
  --hidden_size INTEGER     Hidden layer size [default: 128]
  --channel_n INTEGER       Number of channels [default: 16]
  --save_every INTEGER      Save checkpoint every N steps [default: 2000]
  --viz_every INTEGER       Visualize every N steps [default: 2000]
  --device TEXT            Device (cuda/cpu) [default: auto]
```

### Visualization Options

```bash
python scripts/visualize.py [OPTIONS]

Options:
  --checkpoint TEXT         Path to model checkpoint
  --emoji TEXT             Target emoji
  --seed-emoji TEXT        Seed emoji
  --mode TEXT              Visualization mode (growth/regeneration/grid)
  --steps INTEGER          Number of CA steps [default: 200]
```

---

## Troubleshooting

### Common Issues

**1. CUDA not available**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**2. Training too slow**
- Reduce `--size` to 64 or 40
- Reduce `--steps` to 10000
- Use GPU if available

**3. Out of memory**
- Reduce `--batch_size` to 4 or 2
- Reduce `--size` to 64 or 40
- Close other applications

**4. Loss not decreasing**
- Increase `--steps` to 25000+
- Try different emoji pairs
- Check learning rate

---

## Tips & Best Practices

### For Best Results:
- ✅ Use GPU for training
- ✅ Train for at least 15,000 steps
- ✅ Use 80×80 or higher resolution
- ✅ Monitor loss curve for convergence

### For Quick Testing:
- Use `--size 40` and `--steps 5000`
- Reduce `--viz_every` to 1000
- Test on CPU first

### For Production:
- Train for 20,000+ steps
- Use `--size 80` or `--size 100`
- Save checkpoints frequently
- Generate multiple visualizations

---

## References

- **Original Paper:** [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) (Distill, 2020)
- **PyTorch Documentation:** https://pytorch.org/docs/
- **Project Repository:** (Your GitHub repo here)

---

## Contact

**Student:** Niken Goswami
**Project:** Master's Thesis Pre-Task
**Date:** December 2024

For questions or issues, please refer to the documentation in `docs/` or consult with your supervisor.
