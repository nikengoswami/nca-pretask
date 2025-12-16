# Growing NCA - Quick Start Guide

## What is this project?

This project implements Growing Neural Cellular Automata (NCA) - a fascinating technique where a neural network learns simple local rules that allow cells to grow complex patterns from a single seed, mimicking biological growth!

## Installation

```bash
cd nca_pretask
pip install -r requirements.txt
```

## Quick Training

Train the NCA to grow a lizard emoji (should take ~10-15 minutes on CPU, faster on GPU):

```bash
python train.py --emoji lizard --steps 8000
```

Other emoji options:
- `--emoji lizard` (default)
- `--emoji fire`
- `--emoji crab`
- `--emoji butterfly`
- `--emoji mushroom`
- `--emoji peacock`

Or use your own image:
```bash
python train.py --image path/to/your/image.png --steps 8000
```

## Visualization

After training, visualize the results:

```bash
# Show growth animation
python visualize.py --mode growth

# Test regeneration after damage
python visualize.py --mode regeneration

# Show growth at different time steps
python visualize.py --mode grid

# Run all visualizations
python visualize.py --mode all
```

## Project Structure

```
nca_pretask/
├── models/
│   └── nca.py              # Core NCA model with perception and update rules
├── utils/
│   └── helpers.py          # Utility functions for loading images, visualization
├── train.py                # Training script
├── visualize.py            # Visualization and testing script
├── outputs/                # Generated visualizations
│   ├── target.png          # Target emoji image
│   ├── progress_step_*.png # Training progress visualizations
│   └── final_output.png    # Final trained result
└── checkpoints/            # Saved model weights
    ├── model_latest.pt     # Most recent checkpoint
    └── model_final.pt      # Final trained model
```

## Key Concepts

### 1. **Cell State**
Each cell has 16 channels:
- **Channels 0-2**: RGB color
- **Channel 3**: Alpha (transparency/alive marker)
- **Channels 4-15**: Hidden state (learned features)

### 2. **Perception**
Each cell "sees" its neighbors using Sobel filters (edge detectors):
- Horizontal gradient (dx)
- Vertical gradient (dy)
- Current value

### 3. **Update Rule**
A small neural network (~10k parameters) takes perception as input and outputs how the cell should change.

### 4. **Stochastic Updates**
Only a random subset of cells update at each time step (fire_rate=0.5 by default). This makes the system robust!

### 5. **Living Mask**
Cells with alpha > 0.1 are considered "alive". Dead cells don't update.

## Training Parameters

```bash
python train.py \
  --emoji lizard \           # Target emoji or name
  --size 40 \                # Grid size (40x40)
  --n_channels 16 \          # Channels per cell
  --hidden_size 128 \        # Neural network hidden layer size
  --fire_rate 0.5 \          # Cell update probability
  --steps 8000 \             # Training iterations
  --batch_size 8 \           # Batch size
  --lr 0.002 \               # Learning rate
  --save_every 500 \         # Save checkpoint frequency
  --viz_every 1000           # Visualization frequency
```

## Expected Results

After 8000 training steps, you should see:
- Clean emoji growth from single seed
- Stable final pattern
- Ability to regrow after damage
- Training loss < 0.01

## Troubleshooting

**Out of memory?**
- Reduce `--batch_size` (try 4 or 2)
- Reduce `--size` (try 32 instead of 40)
- Reduce `--hidden_size` (try 64)

**Not converging?**
- Increase `--steps` (try 12000)
- Adjust `--lr` (try 0.001 or 0.003)
- Check that target image loaded correctly in `outputs/target.png`

**Training too slow?**
- Make sure you have a GPU with CUDA installed
- Check `python train.py` output - it should say "Using device: cuda"
- If CPU only, reduce `--size` to 32 for faster training

## Next Steps (for BNN-NCA Thesis)

This pre-task gives you the foundation for the main thesis work:

1. **Task 1**: Convert this deterministic NCA to a **Bayesian NCA**
   - Replace Conv2d layers with Bayesian versions
   - Add uncertainty quantification
   - Train with variational inference

2. **Task 2a**: Use uncertainty for **OoD detection**
   - Test on novel/corrupted patterns
   - Measure epistemic uncertainty

3. **Task 2b**: Apply to **Federated Learning**
   - Detect failing/malicious clients using uncertainty
   - Robust aggregation

## Helpful Resources

- Original paper: https://distill.pub/2020/growing-ca/
- BNN resources: (papers Nick sent)
- PyTorch documentation: https://pytorch.org/docs/

Good luck with your thesis!
