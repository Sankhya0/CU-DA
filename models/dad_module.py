# models/dad_module.py

import torch
import torch.nn as nn

class DiffusionOperator(nn.Module):
    def __init__(self, beta):
        super(DiffusionOperator, self).__init__()
        self.beta = beta

    def forward(self, x):
        noise = torch.randn_like(x) * self.beta
        return x + noise

class ReverseOperator(nn.Module):
    def __init__(self):
        super(ReverseOperator, self).__init__()
        # Add Global Average Pooling to reduce input size to (batch_size, 2048)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.network = nn.Sequential(
            nn.Linear(2048, 2048),  # Linear layer expects input of size 2048
            nn.ReLU(),
            nn.Linear(2048, 2048)
        )

    def forward(self, x):
        # Apply Global Average Pooling to reduce (batch_size, 2048, 7, 7) to (batch_size, 2048)
        x = self.gap(x)  # Output shape: (batch_size, 2048, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 2048)
        return self.network(x)

class DADModule(nn.Module):
    def __init__(self, steps, beta):
        super(DADModule, self).__init__()
        self.steps = steps
        self.diffusion_op = DiffusionOperator(beta)
        self.reverse_op = ReverseOperator()

    def forward(self, x):
        features = x
        all_steps = []

        # Diffusion process: add noise iteratively
        for _ in range(self.steps):
            features = self.diffusion_op(features)
            all_steps.append(features)

        # Reverse process using the reverse operator
        features = self.reverse_op(features)

        return features, all_steps
