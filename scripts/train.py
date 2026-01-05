"""EXTREME QUALITY training - maximum steps and resolution for best results."""

import argparse
import os
from pathlib import Path
import time

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
    """Compute loss between current state and target."""
    return F.mse_loss(x[:, :4], target[:, :4])


def train_step(model, optimizer, seed, target, pool, batch_size=8):
    """Single training step."""
    # Sample from pool or use seed
    batch = pool.sample(batch_size)
    if batch is None:
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

    # Gradient normalization - crucial for stability
    for param in model.parameters():
        if param.grad is not None:
            param.grad = param.grad / (param.grad.norm() + 1e-8)

    optimizer.step()

    # Commit batch to pool
    pool.commit(x.detach())

    return loss.item()


def validate(model, seed, target, n_steps=200):
    """Validation: grow from seed for fixed number of steps."""
    with torch.no_grad():
        x = seed.to(model.device)

        for _ in range(n_steps):
            x = model.update(x)

        loss = loss_fn(x, target.to(model.device))

    return x, loss.item()


def visualize_progress(model, seed, target, step, output_dir, n_steps=200, n_frames=21):
    """Visualize growth sequence at current training step."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        x = seed.to(model.device)

        # Collect frames
        frames = [to_rgb(x.cpu())[0].permute(1, 2, 0).numpy()]

        step_interval = n_steps // (n_frames - 1)

        for i in range(n_steps):
            x = model.update(x)
            if (i + 1) % step_interval == 0:
                frames.append(to_rgb(x.cpu())[0].permute(1, 2, 0).numpy())

        # Plot with higher DPI
        fig, axes = plt.subplots(1, n_frames + 1, figsize=(2.5 * (n_frames + 1), 2.5))

        for i, frame in enumerate(frames):
            axes[i].imshow(frame)
            axes[i].set_title(f'Step {i * step_interval}', fontsize=10)
            axes[i].axis('off')

        # Show target
        axes[-1].imshow(to_rgb(target.cpu())[0].permute(1, 2, 0).numpy())
        axes[-1].set_title('Target', fontsize=10, fontweight='bold')
        axes[-1].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / f'progress_step_{step}.png', dpi=200, bbox_inches='tight')
        plt.close()

    # Also save final output separately
    save_tensor_as_image(to_rgb(x.cpu())[0], output_dir / f'final_step_{step}.png')


def main():
    parser = argparse.ArgumentParser(description='Train EXTREME QUALITY NCA model')
    parser.add_argument('--emoji', type=str, default='mushroom',
                        help='Target emoji name')
    parser.add_argument('--seed-emoji', type=str, default='lizard',
                        help='Seed emoji name')
    parser.add_argument('--size', type=int, default=80,
                        help='Image size (80 for extreme quality)')
    parser.add_argument('--steps', type=int, default=20000,
                        help='Training steps (20000 for extreme quality)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden layer size')
    parser.add_argument('--channel_n', type=int, default=16,
                        help='Number of channels (4 RGBA + 12 hidden)')
    parser.add_argument('--pool_size', type=int, default=1024,
                        help='Sample pool size')
    parser.add_argument('--save_every', type=int, default=2000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--viz_every', type=int, default=2000,
                        help='Visualize progress every N steps')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu, auto-detect if not specified)')

    args = parser.parse_args()

    # Setup device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("EXTREME QUALITY NCA TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    print("Configuration:")
    print(f"  Resolution: {args.size}x{args.size} (was 40x40)")
    print(f"  Training steps: {args.steps:,} (was 8,000)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Hidden size: {args.hidden_size}")
    print()

    # Estimate time
    if device.type == 'cuda':
        est_time = "15-30 minutes"
    else:
        est_time = "5-8 hours"
    print(f"Estimated training time: {est_time}")
    print("=" * 60)
    print()

    # Create output directories
    output_dir = Path('outputs_extreme')
    checkpoint_dir = Path('checkpoints_extreme')
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    # Load images
    print(f"Loading emojis...")
    print(f"  Seed: {args.seed_emoji}")
    print(f"  Target: {args.emoji}")
    print(f"  Size: {args.size}x{args.size}")

    seed_img = load_emoji(args.seed_emoji, size=args.size)
    target_img = load_emoji(args.emoji, size=args.size)

    # Save seed and target
    save_tensor_as_image(to_rgb(seed_img)[0], output_dir / 'seed.png')
    save_tensor_as_image(to_rgb(target_img)[0], output_dir / 'target.png')

    print("OK Emojis loaded and saved")

    # Create seed from emoji (for emoji-to-emoji transformation)
    seed = torch.zeros(1, args.channel_n, args.size, args.size, device=device)
    seed[:, :4, :, :] = seed_img  # Copy RGBA channels

    # Create target
    target = torch.zeros(1, args.channel_n, args.size, args.size, device=device)
    target[:, :4, :, :] = target_img  # Copy RGBA channels

    # Create model
    print(f"\nCreating model...")
    model = create_model(
        n_channels=args.channel_n,
        fire_rate=0.5,
        hidden_size=args.hidden_size,
        device=device
    )

    print_model_summary(model)

    # Create optimizer and pool
    optimizer = Adam(model.parameters(), lr=args.lr)
    pool = SamplePool(size=args.pool_size)

    # Training loop
    print(f"\nStarting EXTREME QUALITY training...")
    print(f"This will take approximately: {est_time}")
    print()

    start_time = time.time()
    losses = []

    pbar = tqdm(range(args.steps), desc='Training', ncols=100)

    for step in pbar:
        # Training step
        loss = train_step(model, optimizer, seed, target, pool, args.batch_size)
        losses.append(loss)

        # Update progress bar with more info
        if step > 0 and step % 100 == 0:
            avg_loss_100 = sum(losses[-100:]) / min(100, len(losses))
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed
            eta = (args.steps - step - 1) / steps_per_sec / 60  # minutes

            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'avg100': f'{avg_loss_100:.4f}',
                'eta': f'{eta:.1f}m'
            })
        else:
            pbar.set_postfix({'loss': f'{loss:.4f}'})

        # Save checkpoint
        if (step + 1) % args.save_every == 0:
            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'losses': losses,
            }, checkpoint_dir / f'model_step_{step + 1}.pt')

        # Visualize progress
        if (step + 1) % args.viz_every == 0:
            print(f"\n  -> Generating visualization at step {step + 1}...")
            visualize_progress(model, seed, target, step + 1, output_dir)
            print(f"  OK Saved to {output_dir / f'progress_step_{step + 1}.png'}")

    # Final visualization
    print("\n\nGenerating final visualization...")
    visualize_progress(model, seed, target, args.steps, output_dir)

    # Save final model
    torch.save({
        'step': args.steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
    }, checkpoint_dir / 'model_final.pt')

    # Save loss plot
    print("Creating loss plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(losses, alpha=0.3, label='Loss')
    # Moving average
    window = 100
    if len(losses) >= window:
        moving_avg = [sum(losses[max(0, i-window):i+1]) / min(window, i+1)
                     for i in range(len(losses))]
        plt.plot(moving_avg, linewidth=2, label=f'Moving Average ({window} steps)')

    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    total_time = time.time() - start_time

    print()
    print("=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Total training time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Loss reduction: {(1 - losses[-1]/losses[0])*100:.1f}%")
    print()
    print(f"Outputs saved to: {output_dir.absolute()}")
    print(f"Checkpoints saved to: {checkpoint_dir.absolute()}")
    print()
    print("Key files to check:")
    print(f"  -> {output_dir / f'progress_step_{args.steps}.png'}")
    print(f"  -> {output_dir / f'final_step_{args.steps}.png'}")
    print(f"  -> {output_dir / 'loss_curve.png'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
