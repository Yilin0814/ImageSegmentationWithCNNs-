import os
import torch
import torch.optim as optim
from tqdm import tqdm  # For a nice progress bar
from model import UNetLext
from utils import dice_loss_from_logits, dice_score_from_logits


class Trainer:
    """
    Manages the model training and validation loop.
    """

    def __init__(self, args, device, train_loader, val_loader):
        self.args = args
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 1. Initialize Model
        # [MODIFIED] Set input_channels=1 to accept grayscale images
        self.model = UNetLext(input_channels=1, output_channels=1).to(self.device)

        # 2. Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.wd
        )

        # 3. Initialize Loss Function (from utils.py)
        # This is the "Soft Dice Loss" for training
        self.criterion = dice_loss_from_logits

        # 4. Create output directory for saving the best model
        os.makedirs(self.args.out_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.args.out_dir, 'best_model.pth')
        print(f"Best model will be saved to: {self.best_model_path}")

        # 5. Track the best validation score
        self.best_val_dice = 0.0

    def train_epoch(self, epoch):
        """
        Executes one full training epoch.
        """
        self.model.train()  # Set model to training mode
        total_loss = 0.0

        # Use tqdm for a progress bar
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{int(self.args.epochs)} [Training]")

        for images, masks in loop:
            # Move data to the selected device (GPU/CPU)
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            # 1. Forward pass
            logits = self.model(images)

            # 2. Calculate loss
            loss = self.criterion(logits, masks)

            # 3. Backward pass and optimization
            self.optimizer.zero_grad()  # Clear old gradients
            loss.backward()  # Calculate new gradients
            self.optimizer.step()  # Update model weights

            total_loss += loss.item()

            # Update the progress bar description
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch + 1} Training Loss: {avg_loss:.4f}")

    def validate_epoch(self, epoch):
        """
        Executes one full validation epoch.
        """
        self.model.eval()  # Set model to evaluation mode
        total_dice = 0.0

        loop = tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{int(self.args.epochs)} [Validation]")

        with torch.no_grad():  # No gradients needed for validation
            for images, masks in loop:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                # Forward pass
                logits = self.model(images)

                # Calculate "Hard Dice Score" (from utils.py)
                # This is for evaluation, not for training loss
                dice = dice_score_from_logits(logits, masks)
                total_dice += dice

                loop.set_postfix(dice=f"{dice:.4f}")

        # Calculate average Dice score for the epoch
        avg_dice = total_dice / len(self.val_loader)
        print(f"Epoch {epoch + 1} Validation Dice Score: {avg_dice:.4f}")

        # Save the model if it's the best one seen so far
        if avg_dice > self.best_val_dice:
            self.best_val_dice = avg_dice
            torch.save(self.model.state_dict(), self.best_model_path)
            print(f"New best model saved with Dice Score: {self.best_val_dice:.4f}")

    def run(self):
        """
        Runs the complete training and validation loop.
        """
        print(f"Training started for {int(self.args.epochs)} epochs.")

        for epoch in range(int(self.args.epochs)):
            self.train_epoch(epoch)
            self.validate_epoch(epoch)

        print(
            f"Training complete. Best model saved to {self.best_model_path} with Dice Score: {self.best_val_dice:.4f}")