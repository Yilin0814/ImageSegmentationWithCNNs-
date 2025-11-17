import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# --- 1. Define Paths and Configuration ---
PROJECT_DIR = '/Users/wangyilin/PycharmProjects/MachineVisionAndCNN/final_assignment'
XRAYS_SUBDIR = 'xrays'
MASKS_SUBDIR = 'masks'
# Relative path for all CSV outputs
CSVS_SUBDIR = 'CSVs'
FULL_MAP_CSV = 'data_map.csv'

# Full paths
XRAYS_DIR = os.path.join(PROJECT_DIR, XRAYS_SUBDIR)
MASKS_DIR = os.path.join(PROJECT_DIR, MASKS_SUBDIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, CSVS_SUBDIR)
FULL_MAP_PATH = os.path.join(OUTPUT_DIR, FULL_MAP_CSV)

# Split configuration
TRAIN_RATIO = 0.8  # 80% for training
VAL_RATIO = 0.2  # 20% for validation
RANDOM_SEED = 42  # For reproducible splits

print("--- Starting Dataset Generation Process ---")


# --- Step 1: Create the Full Data Map (data_map.csv) ---

def create_full_data_map(xrays_dir, masks_dir, xray_subdir, mask_subdir):
    """Scans directories to create a DataFrame mapping xrays to masks."""

    # Check if directories exist
    if not os.path.isdir(xrays_dir):
        print(f"❌ Error: X-ray directory not found at {xrays_dir}")
        return None
    # We allow masks_dir to be missing, treating all as test images if so

    # 1. Read X-rays and map by index
    xrays_map = {}
    for filename in os.listdir(xrays_dir):
        if filename.endswith('.png'):
            try:
                # Extract index from filename (e.g., "149.png" -> 149)
                index = int(filename.split('.')[0])
                # Store path relative to the PROJECT_DIR
                xrays_map[index] = os.path.join(xray_subdir, filename)
            except ValueError:
                continue

    # 2. Read Masks and map by index
    masks_map = {}
    if os.path.isdir(masks_dir):
        for filename in os.listdir(masks_dir):
            if filename.endswith('.png'):
                try:
                    index = int(filename.split('.')[0])
                    # Store path relative to the PROJECT_DIR
                    masks_map[index] = os.path.join(mask_subdir, filename)
                except ValueError:
                    continue

    # 3. Build the combined data list
    all_indices = sorted(xrays_map.keys())
    data = {'xrays': [], 'masks': []}

    for index in all_indices:
        data['xrays'].append(xrays_map[index])
        # Get mask path, default to None if not found
        data['masks'].append(masks_map.get(index, None))

    df = pd.DataFrame(data)
    print(f"✅ Data map DataFrame created with {len(df)} entries.")
    return df


# Run the function to create the full map
df_full = create_full_data_map(XRAYS_DIR, MASKS_DIR, XRAYS_SUBDIR, MASKS_SUBDIR)

if df_full is None:
    exit()

# Ensure output directory exists and save the full map
os.makedirs(OUTPUT_DIR, exist_ok=True)
df_full.to_csv(FULL_MAP_PATH, index=False)
print(f"✅ Full data map saved to {FULL_MAP_PATH}")


# --- Step 2: Split the Dataset and Create Train/Val/Test CSVs ---

def split_and_save_csvs(df, train_ratio, val_ratio, output_dir, seed):
    """Splits the dataset into train, validation, and test sets and saves CSVs."""

    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # A. Split into Test Set (masks is None) and Labeled Set (masks is not None)
    df_test = df[df['masks'].isna()].copy()
    df_labeled = df[df['masks'].notna()].copy()

    # B. Split Labeled Set into Train and Validation
    labeled_indices = df_labeled.index.tolist()
    # Shuffle the indices
    random.shuffle(labeled_indices)

    # Calculate split point
    train_count = int(len(labeled_indices) * train_ratio)

    # Get indices for train and validation
    train_indices = labeled_indices[:train_count]
    val_indices = labeled_indices[train_count:]

    # Create DataFrames
    df_train = df_labeled.loc[train_indices]
    df_val = df_labeled.loc[val_indices]

    # C. Save the three CSV files
    df_train.to_csv(os.path.join(output_dir, 'train_set.csv'), index=False)
    df_val.to_csv(os.path.join(output_dir, 'val_set.csv'), index=False)
    df_test.to_csv(os.path.join(output_dir, 'test_set.csv'), index=False)

    print("-" * 40)
    print("✨ Dataset Split Summary:")
    print(f"  - Labeled Samples: {len(df_labeled)}")
    print(f"  - Train Set (80%): {len(df_train)} images saved to train_set.csv")
    print(f"  - Validation Set (20%): {len(df_val)} images saved to val_set.csv")
    print(f"  - Test Set (Masks=None): {len(df_test)} images saved to test_set.csv")

    return len(df_train), len(df_val), len(df_test)


# Run the split function
train_count, val_count, test_count = split_and_save_csvs(df_full, TRAIN_RATIO, VAL_RATIO, OUTPUT_DIR, RANDOM_SEED)


# --- Step 3: Verify CSVs with a Bar Chart ---

def create_verification_chart(train_c, val_c, test_c, output_dir):
    """Generates and saves a bar chart verifying the split."""

    counts = {
        'Train': train_c,
        'Validation': val_c,
        'Test': test_c
    }

    names = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(7, 5))
    bars = plt.bar(names, values, color=['#4CAF50', '#FFC107', '#2196F3'])

    # Add data labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, yval, ha='center', va='bottom', fontsize=10)

    plt.title('Dataset Split Verification', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xlabel('Dataset Type', fontsize=12)
    plt.ylim(0, max(values) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    chart_path = os.path.join(output_dir, 'dataset_split_chart.png')
    plt.savefig(chart_path)

    print("-" * 40)
    print(f"📊 Verification chart saved to: {chart_path}")
    plt.show()


# Run the chart creation function
create_verification_chart(train_count, val_count, test_count, OUTPUT_DIR)

print("--- Dataset Generation and Verification Completed ---")