# utils.py

import torch

def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth.tar"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)

def accuracy(output, target):
    _, preds = torch.max(output, 1)
    return (preds == target).float().mean().item()
