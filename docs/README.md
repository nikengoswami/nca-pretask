# Growing Neural Cellular Automata - Pre-task

Implementation of Growing Neural Cellular Automata based on the Distill publication "Growing Neural Cellular Automata" (2020).

## Project Structure
```
nca_pretask/
├── models/          # NCA model implementations
├── utils/           # Helper functions and utilities
├── data/            # Target emoji images
├── outputs/         # Generated visualizations
├── checkpoints/     # Saved model weights
├── train.py         # Training script
└── visualize.py     # Visualization and testing script
```

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Train NCA on emoji target
python train.py --emoji lizard --steps 8000

# Visualize growth
python visualize.py --checkpoint checkpoints/model_latest.pt
```

## Goal
Train a neural network to grow an emoji pattern from a single seed cell using only local update rules.
