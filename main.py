import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from models.backbone import build_backbone
from models.classifier import Classifier
from models.dad_module import DADModule
from data.office_home_loader import get_data_loader
from utils import save_checkpoint
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Log all INFO and above messages
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def compute_accuracy(predictions, labels):
    """Computes accuracy for a batch."""
    _, predicted_labels = torch.max(predictions, 1)
    correct = (predicted_labels == labels).sum().item()
    accuracy = correct / labels.size(0) * 100  # Accuracy in percentage
    return accuracy

def train_with_mls():
    try:
        logging.info("Initializing configuration...")
        config = Config()

        # Initialize models
        logging.info("Building backbone...")
        conv_block = build_backbone().to(config.DEVICE)
        logging.info(f"Backbone initialized and moved to {config.DEVICE}")

        logging.info("Initializing DAD module...")
        dad_module = DADModule(config.DIFFUSION_STEPS, beta=0.02).to(config.DEVICE)
        classifier = Classifier().to(config.DEVICE)

        logging.info("Loading datasets...")
        source_loader = get_data_loader(f"{config.DATASET_PATH}/{config.SOURCE_DOMAIN}", config.BATCH_SIZE)
        target_loader = get_data_loader(f"{config.DATASET_PATH}/{config.TARGET_DOMAIN}", config.BATCH_SIZE)

        logging.info(f"Loaded {len(source_loader.dataset)} samples from source domain.")
        logging.info(f"Loaded {len(target_loader.dataset)} samples from target domain.")

        # Set up loss function and optimizers
        criterion = nn.CrossEntropyLoss()
        optimizer_classifier = optim.SGD(
            classifier.parameters(), lr=config.LR, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY
        )
        optimizer_dad = optim.SGD(
            dad_module.parameters(), lr=config.LR, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY
        )

        logging.info("Starting training...")

        for epoch in range(config.EPOCHS):
            logging.info(f"Epoch {epoch + 1}/{config.EPOCHS} started.")

            classifier.train()
            dad_module.train()

            total_correct = 0
            total_samples = 0
            epoch_loss = 0.0

            for batch_idx, (inputs, labels) in enumerate(source_loader):
                logging.debug(f"Processing batch {batch_idx + 1}/{len(source_loader)}...")
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

                # ---- C→D Learning: Train DAD using the classifier ----
                with torch.no_grad():
                    features = conv_block(inputs)

                transformed_features, all_steps = dad_module(features)

                optimizer_dad.zero_grad()
                loss_dad = criterion(classifier(transformed_features), labels)
                loss_dad.backward()
                optimizer_dad.step()

                logging.debug(f"Batch {batch_idx + 1} - Loss (DAD): {loss_dad.item():.4f}")

                # ---- D→C Learning: Fine-tune the classifier ----
                optimizer_classifier.zero_grad()
                total_loss = 0.0  # Accumulate loss across all steps

                for step_idx, step_features in enumerate(all_steps):
                    outputs = classifier(step_features)
                    loss_classifier = criterion(outputs, labels)
                    total_loss += loss_classifier  # Accumulate loss

                    logging.debug(
                        f"Batch {batch_idx + 1}, Step {step_idx + 1} - Loss (Step): {loss_classifier.item():.4f}"
                    )

                # Backpropagate the total accumulated loss
                total_loss.backward()
                optimizer_classifier.step()

                # Update epoch loss and accuracy
                batch_accuracy = compute_accuracy(outputs, labels)
                epoch_loss += total_loss.item()
                total_correct += (outputs.argmax(1) == labels).sum().item()
                total_samples += labels.size(0)

                logging.info(
                    f"Batch {batch_idx + 1}/{len(source_loader)} - "
                    f"Loss: {total_loss.item():.4f}, Accuracy: {batch_accuracy:.2f}%"
                )

            # Compute epoch-level accuracy and average loss
            epoch_accuracy = (total_correct / total_samples) * 100
            average_loss = epoch_loss / len(source_loader)

            logging.info(
                f"Epoch {epoch + 1} completed. "
                f"Average Loss: {average_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%"
            )

            # Save checkpoint every 50 epochs
            if (epoch + 1) % 50 == 0:
                save_checkpoint(classifier, optimizer_classifier, epoch + 1)
                logging.info(f"Checkpoint saved at epoch {epoch + 1}.")

        logging.info("Training complete.")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    logging.info("Starting the DAD training script...")
    train_with_mls()
