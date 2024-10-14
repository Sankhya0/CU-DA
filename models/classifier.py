# models/classifier.py

import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, num_classes=65):
        super(Classifier, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.fc = nn.Linear(2048, num_classes)  # Linear layer

    def forward(self, x):
        # If input is 4D (batch_size, channels, height, width), apply GAP
        if len(x.shape) == 4:
            x = self.gap(x)  # Reduce to (batch_size, 2048, 1, 1)
            x = x.view(x.size(0), -1)  # Flatten to (batch_size, 2048)
        elif len(x.shape) == 2:
            # If input is already 2D, no need to apply GAP
            pass
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        return self.fc(x)
