# models/backbone.py

import torch.nn as nn
import torchvision.models as models

def build_backbone():
    # Load the ResNet50 model with correct weights
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Select only up to layer4 (removing fully connected layers)
    layers = list(backbone.children())[:-2]  # Remove avgpool and fc layers
    conv_block = nn.Sequential(*layers)
    return conv_block
