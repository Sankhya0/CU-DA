import torch

class Config:
    # Paths
    DATASET_PATH = "C:/Users/dixit/OneDrive/Desktop/OfficeHome"
    SOURCE_DOMAIN = "Art"
    TARGET_DOMAIN = "Clipart"

    # Training hyperparameters
    BATCH_SIZE = 2  # Reduce to 2
    EPOCHS = 10      # Run only 1 epoch
    LR = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.05
    POLY_POWER = 0.9

    # DAD-specific hyperparameters
    DIFFUSION_STEPS = 20  # Reduce to 10 steps
    REVERSE_ITERATIONS = 4  # Reduce to 2 reverse iterations

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# class Config:
#     # Paths
#     DATASET_PATH = "C:/Users/dixit/OneDrive/Desktop/OfficeHome"
#     SOURCE_DOMAIN = "Art"
#     TARGET_DOMAIN = "Clipart"

#     # Training hyperparameters
#     BATCH_SIZE = 8
#     EPOCHS = 300
#     LR = 0.001
#     MOMENTUM = 0.9
#     WEIGHT_DECAY = 0.05
#     POLY_POWER = 0.9

#     # DAD-specific hyperparameters
#     DIFFUSION_STEPS = 600
#     REVERSE_ITERATIONS = 20

#     # Device
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
