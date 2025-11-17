# Bone Structure Segmentation with U-Net

This project implements a deep learning pipeline for the automatic semantic segmentation of bone structures (tibia or femur) from X-ray images. It uses a custom U-Net architecture (`UNetLext`) trained on manually annotated data.

## 📌 Project Overview

The goal of this project is to accurately generate binary masks for specific bone structures. The pipeline includes data loading, preprocessing (grayscale conversion), model training, validation (using Dice Score), and final prediction on an unlabeled test set.

### Key Features
* **Model:** Custom U-Net architecture (`UNetLext`) with dynamic depth and width.
* **Input:** 1-Channel Grayscale X-ray images ($256 \times 256$).
* **Metric:** Dice Score (Soft Dice Loss for training, Hard Dice for evaluation).
* **Dataset:**
    * **Training/Validation:** 100 labeled images.
    * **Test:** 50 unlabeled images (used for final mask prediction).

## 📂 Project Structure

The project follows a modular structure to ensure clean code and reproducibility:

```text
.
├── args.py           # Configuration parameters (epochs, batch_size, paths)
├── dataset.py        # Custom Dataset class (handles image/mask loading & transforms)
├── evaluate.py       # Evaluates model on Validation set & generates predictions for Test set
├── main.py           # Entry point: sets up data loaders and starts the Trainer
├── model.py          # Definition of the UNetLext architecture
├── trainer.py        # Manages the training loop, validation loop, and model saving
├── utils.py          # Helper functions for Dice Loss and Dice Score calculations
├── xrays/            # Folder containing X-ray images
├── masks/            # Folder containing Ground Truth masks
├── CSVs/             # CSV files defining the data splits (train_set, val_set, test_set)
└── session/          # Output directory for saved models and predictions
    ├── best_model.pth
    └── predictions/

```
https://github.com/Yilin0814/ImageSegmentationWithCNNs-/blob/main/Snipaste_2025-11-16_20-43-07.jpg?raw=true<img width="942" height="536" alt="image" src="https://github.com/user-attachments/assets/e4bba5c5-c5d2-4eb8-8c7f-9def18c668bb" />


## 🚀 Installation & Requirements
Ensure you have Python installed. You can install the necessary dependencies using:

```
pip install torch torchvision pandas pillow tqdm
```

## ⚙️ Usage
1. Training the Model
To start the training process, run main.py. This script initializes the BoneDataset, sets up the UNetLext model, and triggers the Trainer.

```
python main.py
```

During training, the best model (highest Validation Dice Score) is automatically saved to session/best_model.pth.

2. Evaluation & Prediction
After training, use evaluate.py to perform the final evaluation. This script performs two critical tasks:

Validation Score: Calculates and reports the Dice Score on the validation set (Ground Truth available).

Test Prediction: Generates binary masks for the 50 unlabeled test images.

``` Bash

python evaluate.py
```
Output:

The script prints the final Validation Dice Score to the console.

Predicted masks are saved as .png files in session/predictions/.

## 🧠 Model Details
The architecture is a U-Net variant defined in model.py.

Input Channels: 1 (Grayscale).

Output Channels: 1 (Binary Mask).

Loss Function: Soft Dice Loss (optimized to maximize overlap).

Optimization: AdamW optimizer.

## 📊 Results
Metric: Dice Score (Range: 0.0 to 1.0).

Best Validation Score: 0.9580

