"""Training script for Growing Neural Cellular Automata."""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.nca import create_model, SamplePool
from utils.helpers import (
    load_emoji, make_seed, to_rgb,
    print_model_summary, save_tensor_as_image
)


def loss_fn(x, target):
    """
    Compute loss between current state and target.

    Uses mean squared error on RGBA channels only.
    """
    return F.mse_loss(x[:, :4], target[:, :4])


def train_step(model, optimizer, seed, target, pool, batch_size=8):
    """
    Single training step.

    Args:
        model: NCA model
        optimizer: Optimizer
        seed: Initial seed state
        target: Target pattern
        pool: Sample pool
        batch_size: Batch size

    Returns:
        Loss value
    """
    # Sample from pool or use seed
    batch = pool.sample(batch_size)
    if batch is None:
        # Pool not full yet, use seed
        x = seed.repeat(batch_size, 1, 1, 1).to(model.device)
    else:
        x = batch.to(model.device)

    # Random number of steps (64-96 for diversity)
    n_steps = torch.randint(64, 97, (1,)).item()

    # Forward pass
    optimizer.zero_grad()

    for _ in range(n_steps):
        x = model.update(x)

    # Compute loss
    target_batch = target.repeat(batch_size, 1, 1, 1).to(model.device)
    loss = loss_fn(x, target_batch)

    # Overflow loss: penalize cells that grow too large
    overflow_loss = (x - x.clamp(-2.0, 2.0)).abs().sum() * 0.01
    total_loss = loss + overflow_loss

    # Backward pass
    total_loss.backward()

    # Gradient normalization (per-variable L2 normalization) - from original paper
    # This is crucial for training with zero-initialized weights
    for param in model.parameters():
        if param.grad is not None:
            param.grad = param.grad / (param.grad.norm() + 1e-8)

    optimizer.step()

    # Commit batch to pool
    pool.commit(x.detach())

    return loss.item()


def validate(model, seed, target, n_steps=200):
    """
    Validation: grow from seed for fixed number of steps.

    Returns final state and loss.
    """
    with torch.no_grad():
        x = seed.to(model.device)

        for _ in range(n_steps):
            x = model.update(x)

        loss = loss_fn(x, target.to(model.device))

    return x, loss.item()


def save_checkpoint(model, optimizer, step, loss, filepath):
    """Save model checkpoint."""
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=model.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['step'], checkpoint['loss']


def visualize_progress(seed, target, model, output_dir, step):
    """Create and save visualization of current progress."""
    with torch.no_grad():
        x = seed.to(model.device)

        # Grow for 200 steps
        frames = []
        for i in range(200):
            if i % 10 == 0:
                frames.append(to_rgb(x).cpu())
            x = model.update(x)

        # Create figure
        fig, axes = plt.subplots(1, len(frames) + 1, figsize=(20, 3))

        # Show growth frames
        for idx, frame in enumerate(frames):
            axes[idx].imshow(frame[0].permute(1, 2, 0))
            axes[idx].axis('off')
            axes[idx].set_title(f'Step {idx * 10}')

        # Show target
        axes[-1].imshow(to_rgb(target)[0].permute(1, 2, 0).cpu())
        axes[-1].axis('off')
        axes[-1].set_title('Target')

        plt.tight_layout()
        plt.savefig(output_dir / f'progress_step_{step}.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Growing NCA')
    parser.add_argument('--emoji', type=str, default='lizard',
                        help='Emoji name or character (lizard, fire, crab, butterfly, mushroom, peacock)')
    parser.add_argument('--seed-emoji', type=str, default=None,
                        help='Seed emoji name (if None, uses single pixel seed). For emoji-to-emoji transformation.')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to custom target image')
    parser.add_argument('--size', type=int, default=40,
                        help='Grid size (default: 40x40)')
    parser.add_argument('--n_channels', type=int, default=16,
                        help='Number of channels per cell')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden layer size')
    parser.add_argument('--fire_rate', type=float, default=0.5,
                        help='Cell update probability')
    parser.add_argument('--steps', type=int, default=8000,
                        help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='Learning rate')
    parser.add_argument('--pool_size', type=int, default=1024,
                        help='Sample pool size')
    parser.add_argument('--save_every', type=int, default=500,
                        help='Save checkpoint every N steps')
    parser.add_argument('--viz_every', type=int, default=1000,
                        help='Create visualization every N steps')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    output_dir = Path('outputs')
    checkpoint_dir = Path('checkpoints')
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    # Load target
    print(f"\nLoading target...")
    if args.image is not None:
        from utils.helpers import load_image_from_file
        target = load_image_from_file(args.image, size=args.size)
        print(f"Loaded custom image: {args.image}")
    else:
        target = load_emoji(args.emoji, size=args.size)
        print(f"Loaded emoji: {args.emoji}")

    target = target.to(device)

    # Save target image
    save_tensor_as_image(to_rgb(target), output_dir / 'target.png')
    print(f"Target saved to {output_dir / 'target.png'}")

    # Create seed
    if args.seed_emoji is not None:
        # Use emoji as seed for emoji-to-emoji transformation
        print(f"Loading seed emoji: {args.seed_emoji}")
        seed_img = load_emoji(args.seed_emoji, size=args.size).to(device)
        # Pad seed image to have n_channels (RGBA + hidden channels)
        seed = torch.zeros(1, args.n_channels, args.size, args.size, device=device)
        seed[:, :4, :, :] = seed_img  # Copy RGBA channels
        save_tensor_as_image(to_rgb(seed), output_dir / 'seed.png')
        print(f"Seed emoji saved to {output_dir / 'seed.png'}")
    else:
        # Use single pixel seed (original behavior)
        seed = make_seed((1, args.size, args.size), n_channels=args.n_channels)

    # Create model
    print(f"\nCreating model...")
    model = create_model(
        n_channels=args.n_channels,
        hidden_size=args.hidden_size,
        fire_rate=args.fire_rate,
        device=device
    )
    print_model_summary(model)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Sample pool
    pool = SamplePool(size=args.pool_size)

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume is not None:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_step, prev_loss = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from step {start_step}, previous loss: {prev_loss:.6f}")

    # Training loop
    print(f"\nStarting training for {args.steps} steps...")
    print(f"Batch size: {args.batch_size}, Learning rate: {args.lr}")
    print("-" * 60)

    progress_bar = tqdm(range(start_step, args.steps), initial=start_step, total=args.steps)

    for step in progress_bar:
        # Training step
        loss = train_step(model, optimizer, seed, target, pool, batch_size=args.batch_size)

        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss:.6f}'})

        # Validation
        if (step + 1) % 100 == 0:
            val_x, val_loss = validate(model, seed, target, n_steps=200)
            progress_bar.set_postfix({'train_loss': f'{loss:.6f}', 'val_loss': f'{val_loss:.6f}'})

        # Save checkpoint
        if (step + 1) % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f'model_step_{step+1}.pt'
            save_checkpoint(model, optimizer, step + 1, loss, checkpoint_path)

            # Also save as "latest"
            latest_path = checkpoint_dir / 'model_latest.pt'
            save_checkpoint(model, optimizer, step + 1, loss, latest_path)

        # Visualize progress
        if (step + 1) % args.viz_every == 0:
            visualize_progress(seed, target, model, output_dir, step + 1)

    # Final save
    print("\n" + "=" * 60)
    print("Training complete!")
    final_checkpoint = checkpoint_dir / 'model_final.pt'
    save_checkpoint(model, optimizer, args.steps, loss, final_checkpoint)
    print(f"Final model saved to {final_checkpoint}")

    # Final visualization
    print("\nGenerating final visualization...")
    visualize_progress(seed, target, model, output_dir, args.steps)

    # Final validation
    val_x, val_loss = validate(model, seed, target, n_steps=200)
    save_tensor_as_image(to_rgb(val_x), output_dir / 'final_output.png')
    print(f"Final validation loss: {val_loss:.6f}")
    print(f"Final output saved to {output_dir / 'final_output.png'}")


if __name__ == '__main__':
    main()
