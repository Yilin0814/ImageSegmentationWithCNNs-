import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import pandas as pd

from args import get_args
from model import UNetLext
from dataset import BoneDataset
from utils import dice_score_from_logits  # Added back for validation


def evaluate_and_predict():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    model_path = os.path.join(args.out_dir, 'best_model.pth')
    print(f"Model path: {model_path}")

    print("Loading trained model...")
    model = UNetLext(input_channels=1, output_channels=1)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device)
    model.eval()

    # --- [NEW] Part 1: Calculate Dice Score on Validation Set ---
    print("\n--- Part 1: Calculating Final Dice Score (on Validation Set) ---")
    val_csv_path = os.path.join(args.csv_dir, 'val_set.csv')
    print(f"Using validation data: {val_csv_path}")

    try:
        val_dataset = BoneDataset(csv_path=val_csv_path, img_dir=None, mask_dir=None)
    except Exception as e:
        print(f"Error loading validation dataset: {e}")
        return

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, os.cpu_count() // 2)
    )

    total_dice = 0.0
    with torch.no_grad():
        loop_val = tqdm(val_loader, desc="Validating Best Model")
        for images, gt_masks in loop_val:
            images = images.to(device)
            gt_masks = gt_masks.to(device)

            logits = model(images)
            dice = dice_score_from_logits(logits, gt_masks)
            total_dice += dice
            loop_val.set_postfix(batch_dice=f"{dice:.4f}")

    avg_dice = total_dice / len(val_loader)
    print(f"Final Validation Dice Score of the best model: {avg_dice:.4f}")

    # --- Part 2: Generate Predictions on Test Set ---
    print("\n--- Part 2: Generating Predictions (on Test Set) ---")
    test_csv_path = os.path.join(args.csv_dir, 'test_set.csv')
    pred_dir = os.path.join(args.out_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    print(f"Test CSV path: {test_csv_path}")
    print(f"Predictions will be saved to: {pred_dir}")

    try:
        test_df = pd.read_csv(test_csv_path)
        image_filenames = [os.path.basename(p) for p in test_df['xrays'].tolist()]
    except FileNotFoundError:
        print(f"Error: Test CSV not found at {test_csv_path}")
        return
    except KeyError:
        print(f"Error: 'xrays' column not found in {test_csv_path}")
        return

    test_dataset = BoneDataset(csv_path=test_csv_path, img_dir=None, mask_dir=None)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, os.cpu_count() // 2)
    )

    print(f"Found {len(test_dataset)} test images for prediction.")

    pred_idx_counter = 0
    with torch.no_grad():
        loop_pred = tqdm(test_loader, desc="Generating Predictions")

        for images, _ in loop_pred:
            images = images.to(device)
            logits = model(images)
            pred_masks = (torch.sigmoid(logits) > 0.5).float()

            for i in range(pred_masks.shape[0]):
                single_mask_tensor = pred_masks[i].squeeze(0)
                mask_np = single_mask_tensor.cpu().numpy()
                mask_img_data = (mask_np * 255).astype('uint8')
                img_pil = Image.fromarray(mask_img_data)

                filename = image_filenames[pred_idx_counter]
                save_path = os.path.join(pred_dir, filename)
                img_pil.save(save_path)

                pred_idx_counter += 1

    print("\n--- Prediction Complete ---")
    print(f"{len(image_filenames)} predicted masks saved to: {pred_dir}")

if __name__ == "__main__":
    evaluate_and_predict()