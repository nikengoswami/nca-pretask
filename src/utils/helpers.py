"""Helper functions for NCA training and visualization."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import requests
from io import BytesIO


def load_emoji(emoji_name="🦎", size=40):
    """
    Load emoji image as target.

    Args:
        emoji_name: Emoji character or name
        size: Target image size (square)

    Returns:
        RGBA tensor of shape [1, 4, size, size] normalized to [0, 1]
    """
    # Map common names to emoji
    emoji_map = {
        "lizard": "🦎",
        "fire": "🔥",
        "crab": "🦀",
        "butterfly": "🦋",
        "mushroom": "🍄",
        "peacock": "🦚"
    }

    if emoji_name in emoji_map:
        emoji_name = emoji_map[emoji_name]

    # Use Twemoji CDN for consistent emoji rendering
    # Convert emoji to codepoint
    codepoint = "-".join([f"{ord(c):x}" for c in emoji_name])
    url = f"https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/{codepoint}.png"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
    except:
        # Fallback: create a simple circle as target
        print(f"Could not load emoji, creating circle target instead")
        img = Image.new('RGBA', (72, 72), (0, 0, 0, 0))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.ellipse([10, 10, 62, 62], fill=(255, 100, 100, 255))

    # Convert to RGBA and resize
    img = img.convert('RGBA')
    img = img.resize((size, size), Image.Resampling.LANCZOS)

    # Convert to tensor [1, 4, H, W] normalized to [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

    # Pre-multiply alpha for proper blending
    rgb, alpha = img_tensor[:, :3], img_tensor[:, 3:4]
    img_tensor = torch.cat([rgb * alpha, alpha], dim=1)

    return img_tensor


def load_image_from_file(filepath, size=40):
    """Load image from file as target."""
    img = Image.open(filepath).convert('RGBA')
    img = img.resize((size, size), Image.Resampling.LANCZOS)

    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

    # Pre-multiply alpha
    rgb, alpha = img_tensor[:, :3], img_tensor[:, 3:4]
    img_tensor = torch.cat([rgb * alpha, alpha], dim=1)

    return img_tensor


def to_rgb(x):
    """
    Convert NCA state to RGB for visualization.

    Args:
        x: Tensor of shape [B, C, H, W] where first 4 channels are RGBA

    Returns:
        RGB tensor [B, 3, H, W]
    """
    rgb, alpha = x[:, :3], x[:, 3:4]
    # Composite over white background
    return torch.clamp(rgb + (1.0 - alpha), 0.0, 1.0)


def make_seed(shape, n_channels=16):
    """
    Create initial seed state: single alive cell in center.

    Args:
        shape: (batch, height, width)
        n_channels: Number of channels in cell state

    Returns:
        Seed tensor of shape [batch, n_channels, height, width]
    """
    batch, h, w = shape
    seed = torch.zeros(batch, n_channels, h, w)

    # Set center cell to be alive (alpha = 1.0)
    seed[:, 3:4, h//2, w//2] = 1.0

    return seed


def get_living_mask(x):
    """
    Get mask of living cells (alpha > 0.1).

    Args:
        x: Cell state tensor [B, C, H, W]

    Returns:
        Boolean mask [B, 1, H, W]
    """
    alpha = x[:, 3:4]
    return F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1


def make_circle_masks(n, h, w):
    """Create random circular masks for training pool."""
    x = np.linspace(-1.0, 1.0, w)[None, None, :]
    y = np.linspace(-1.0, 1.0, h)[None, :, None]

    center = np.random.uniform(-0.5, 0.5, size=[2, n, 1, 1])
    r = np.random.uniform(0.1, 0.4, size=[n, 1, 1])

    x_dist = x - center[0]
    y_dist = y - center[1]
    dist = np.sqrt(x_dist**2 + y_dist**2)

    mask = (dist < r).astype(np.float32)
    return torch.from_numpy(mask)


def save_tensor_as_image(tensor, filepath):
    """Save tensor as image file."""
    # tensor: [C, H, W] or [1, C, H, W]
    if tensor.dim() == 4:
        tensor = tensor[0]

    # Convert to numpy
    img_array = tensor.permute(1, 2, 0).detach().cpu().numpy()
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_array)
    img.save(filepath)


def print_model_summary(model):
    """Print model architecture and parameter count."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model architecture:\n{model}\n")
