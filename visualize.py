"""Visualization and testing script for trained NCA models."""

import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

from models.nca import create_model
from utils.helpers import load_emoji, make_seed, to_rgb, save_tensor_as_image


def load_model_from_checkpoint(checkpoint_path, n_channels=16, hidden_size=128, fire_rate=0.5):
    """Load trained model from checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model(
        n_channels=n_channels,
        hidden_size=hidden_size,
        fire_rate=fire_rate,
        device=device
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from step {checkpoint['step']}, loss: {checkpoint['loss']:.6f}")

    return model


def visualize_growth(model, seed, n_steps=200, save_path=None):
    """
    Visualize NCA growth process.

    Args:
        model: Trained NCA model
        seed: Initial seed state
        n_steps: Number of growth steps
        save_path: Path to save animation (optional)
    """
    print(f"Simulating growth for {n_steps} steps...")

    with torch.no_grad():
        x = seed.to(model.device)
        frames = []

        for i in range(n_steps):
            # Save frame
            rgb = to_rgb(x).cpu().numpy()[0].transpose(1, 2, 0)
            frames.append(rgb)

            # Update
            x = model.update(x)

    # Create animation
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')

    im = ax.imshow(frames[0], interpolation='nearest')

    def update(frame_idx):
        im.set_array(frames[frame_idx])
        ax.set_title(f'Step {frame_idx}/{n_steps}')
        return [im]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=50, blit=True, repeat=True
    )

    if save_path:
        print(f"Saving animation to {save_path}...")
        writer = animation.PillowWriter(fps=20)
        anim.save(save_path, writer=writer)
        print(f"Animation saved!")

    plt.show()


def test_regeneration(model, seed, n_steps=200, damage_step=100, save_dir=None):
    """
    Test NCA's ability to regenerate after damage.

    Args:
        model: Trained NCA model
        seed: Initial seed state
        n_steps: Total number of steps
        damage_step: Step at which to apply damage
        save_dir: Directory to save results
    """
    print(f"Testing regeneration (damage at step {damage_step})...")

    with torch.no_grad():
        x = seed.to(model.device)
        frames = []

        for i in range(n_steps):
            # Apply damage at specified step
            if i == damage_step:
                # Create circular damage mask
                h, w = x.shape[2], x.shape[3]
                y, x_coords = torch.meshgrid(
                    torch.linspace(-1, 1, h),
                    torch.linspace(-1, 1, w),
                    indexing='ij'
                )
                center_x, center_y = 0.3, 0.3
                radius = 0.3
                damage_mask = ((x_coords - center_x)**2 + (y - center_y)**2) > radius**2
                damage_mask = damage_mask.float().to(model.device)

                x = x * damage_mask.view(1, 1, h, w)
                print(f"  Applied damage at step {i}")

            # Save frame every 5 steps
            if i % 5 == 0:
                rgb = to_rgb(x).cpu().numpy()[0].transpose(1, 2, 0)
                frames.append((i, rgb))

            # Update
            x = model.update(x)

    # Create visualization
    n_frames = len(frames)
    cols = 8
    rows = (n_frames + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    for idx, (step_num, frame) in enumerate(frames):
        axes[idx].imshow(frame, interpolation='nearest')
        axes[idx].axis('off')
        title = f'Step {step_num}'
        if step_num == damage_step:
            title += ' (DAMAGE)'
        axes[idx].set_title(title, fontsize=8)

    # Hide unused subplots
    for idx in range(len(frames), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / 'regeneration_test.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Regeneration test saved to {save_path}")

    plt.show()


def create_growth_grid(model, seed, steps_list=[0, 20, 40, 60, 80, 100, 150, 200], save_path=None):
    """
    Create grid showing growth at different time steps.

    Args:
        model: Trained NCA model
        seed: Initial seed state
        steps_list: List of step numbers to visualize
        save_path: Path to save result
    """
    print(f"Creating growth grid for steps: {steps_list}")

    with torch.no_grad():
        x = seed.to(model.device)
        snapshots = {}

        max_steps = max(steps_list)
        for i in range(max_steps + 1):
            if i in steps_list:
                rgb = to_rgb(x).cpu().numpy()[0].transpose(1, 2, 0)
                snapshots[i] = rgb

            x = model.update(x)

    # Create grid
    n_images = len(steps_list)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, step in enumerate(steps_list):
        row = idx // cols
        col = idx % cols

        axes[row, col].imshow(snapshots[step], interpolation='nearest')
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Step {step}', fontsize=12)

    # Hide unused subplots
    for idx in range(len(steps_list), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Growth grid saved to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize trained NCA')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model_latest.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--emoji', type=str, default='lizard',
                        help='Emoji used during training')
    parser.add_argument('--size', type=int, default=40,
                        help='Grid size')
    parser.add_argument('--n_channels', type=int, default=16,
                        help='Number of channels')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden layer size')
    parser.add_argument('--fire_rate', type=float, default=0.5,
                        help='Fire rate')
    parser.add_argument('--steps', type=int, default=200,
                        help='Number of growth steps to visualize')
    parser.add_argument('--mode', type=str, default='growth',
                        choices=['growth', 'regeneration', 'grid', 'all'],
                        help='Visualization mode')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for visualizations')
    parser.add_argument('--save_animation', action='store_true',
                        help='Save animation as GIF')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model_from_checkpoint(
        args.checkpoint,
        n_channels=args.n_channels,
        hidden_size=args.hidden_size,
        fire_rate=args.fire_rate
    )

    # Create seed
    seed = make_seed((1, args.size, args.size), n_channels=args.n_channels)

    # Run visualizations based on mode
    if args.mode == 'growth' or args.mode == 'all':
        print("\n" + "="*60)
        print("GROWTH VISUALIZATION")
        print("="*60)
        save_path = output_dir / 'growth_animation.gif' if args.save_animation else None
        visualize_growth(model, seed, n_steps=args.steps, save_path=save_path)

    if args.mode == 'regeneration' or args.mode == 'all':
        print("\n" + "="*60)
        print("REGENERATION TEST")
        print("="*60)
        test_regeneration(model, seed, n_steps=args.steps, save_dir=output_dir)

    if args.mode == 'grid' or args.mode == 'all':
        print("\n" + "="*60)
        print("GROWTH GRID")
        print("="*60)
        save_path = output_dir / 'growth_grid.png'
        create_growth_grid(model, seed, save_path=save_path)

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == '__main__':
    main()
