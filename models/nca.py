"""Neural Cellular Automata Model Implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CAModel(nn.Module):
    """
    Neural Cellular Automata Model.

    The model learns local update rules that allow cells to grow complex patterns
    from a single seed cell.
    """

    def __init__(self, n_channels=16, fire_rate=0.5, hidden_size=128, device=None):
        """
        Initialize NCA model.

        Args:
            n_channels: Number of channels per cell (first 4 are RGBA)
            fire_rate: Probability of updating a cell at each step
            hidden_size: Size of hidden layer in update network
            device: Device to run on (cuda/cpu)
        """
        super().__init__()

        self.n_channels = n_channels
        self.fire_rate = fire_rate
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Sobel filters for perception (detect gradients in 3x3 neighborhood)
        self.register_buffer('sobel_x', self._make_sobel_filter('x'))
        self.register_buffer('sobel_y', self._make_sobel_filter('y'))

        # Identity filter to include cell's own state
        identity = torch.zeros(1, 1, 3, 3)
        identity[0, 0, 1, 1] = 1.0
        self.register_buffer('identity', identity)

        # Perception outputs: for each channel, we get [dx, dy, value] = 3 values
        # Total perception size = n_channels * 3
        perception_size = n_channels * 3

        # Update network: perceives local neighborhood and outputs state change
        self.update_net = nn.Sequential(
            nn.Conv2d(perception_size, hidden_size, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, n_channels, 1, bias=False)
        )

        # Initialize last layer to zero for stable training
        with torch.no_grad():
            self.update_net[-1].weight.zero_()

        self.to(self.device)

    def _make_sobel_filter(self, direction):
        """Create 3x3 Sobel filter for gradient detection."""
        if direction == 'x':
            kernel = torch.tensor([[-1.0, 0.0, 1.0],
                                   [-2.0, 0.0, 2.0],
                                   [-1.0, 0.0, 1.0]]) / 8.0
        else:  # y direction
            kernel = torch.tensor([[-1.0, -2.0, -1.0],
                                   [0.0, 0.0, 0.0],
                                   [1.0, 2.0, 1.0]]) / 8.0

        # Shape: [1, 1, 3, 3]
        return kernel.view(1, 1, 3, 3)

    def perceive(self, x):
        """
        Perception: compute local gradients for each channel.

        Args:
            x: Cell states [B, n_channels, H, W]

        Returns:
            Perception tensor [B, n_channels*3, H, W]
        """
        def apply_filter(filter_kernel):
            # Apply same filter to all channels independently
            # Use groups=n_channels for per-channel convolution
            filter_expanded = filter_kernel.repeat(self.n_channels, 1, 1, 1)
            return F.conv2d(x, filter_expanded, padding=1, groups=self.n_channels)

        # Compute gradients and identity for each channel
        dx = apply_filter(self.sobel_x)
        dy = apply_filter(self.sobel_y)
        value = apply_filter(self.identity)

        # Concatenate along channel dimension
        # Output: [B, n_channels*3, H, W]
        perception = torch.cat([dx, dy, value], dim=1)

        return perception

    def update(self, x, fire_rate=None):
        """
        Single update step.

        Args:
            x: Current cell states [B, n_channels, H, W]
            fire_rate: Override default fire rate

        Returns:
            Updated cell states
        """
        # Get perception
        perception = self.perceive(x)

        # Compute state update
        dx = self.update_net(perception)

        # Stochastic update: only update random subset of cells
        if fire_rate is None:
            fire_rate = self.fire_rate

        # Create random mask for stochastic updates
        batch_size, _, h, w = x.shape
        update_mask = (torch.rand(batch_size, 1, h, w, device=self.device) < fire_rate).float()

        # Apply update with mask
        x = x + dx * update_mask

        # Living mask: cells with alpha > 0.1 are alive
        # Use max pooling to consider neighborhood
        alpha = x[:, 3:4, :, :]
        living_mask = F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1

        # Apply living mask
        x = x * living_mask.float()

        return x

    def forward(self, x, steps=1, fire_rate=None):
        """
        Run CA for multiple steps.

        Args:
            x: Initial state [B, n_channels, H, W]
            steps: Number of update steps
            fire_rate: Override fire rate

        Returns:
            Final state after all steps
        """
        for _ in range(steps):
            x = self.update(x, fire_rate)

        return x

    def get_living_mask(self, x):
        """Get mask of living cells."""
        alpha = x[:, 3:4]
        return F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1


class SamplePool:
    """
    Pool of training samples at different growth stages.

    This helps the model learn to grow AND maintain the pattern,
    not just reach the target from the seed.
    """

    def __init__(self, size=1024):
        """
        Initialize sample pool.

        Args:
            size: Maximum number of samples to store
        """
        self.size = size
        self.pool = []

    def sample(self, batch_size):
        """Sample random batch from pool."""
        if len(self.pool) < batch_size:
            return None

        indices = torch.randint(0, len(self.pool), (batch_size,))
        batch = torch.stack([self.pool[i] for i in indices])
        return batch

    def commit(self, batch):
        """Add batch to pool, maintaining maximum size."""
        for sample in batch:
            if len(self.pool) < self.size:
                self.pool.append(sample.detach().cpu())
            else:
                # Replace random sample
                idx = torch.randint(0, self.size, (1,)).item()
                self.pool[idx] = sample.detach().cpu()


def create_model(n_channels=16, hidden_size=128, fire_rate=0.5, device=None):
    """Factory function to create NCA model."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CAModel(
        n_channels=n_channels,
        fire_rate=fire_rate,
        hidden_size=hidden_size,
        device=device
    )

    return model
