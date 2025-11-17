import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 256

img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

mask_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


class BoneDataset(Dataset):
    def __init__(self, csv_path, img_dir, mask_dir):
        try:
            self.df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"Error: CSV file not found at: {csv_path}")
            self.df = pd.DataFrame(columns=['xrays', 'masks'])

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        if 'xrays' not in self.df.columns:
            print(f"Warning: CSV {csv_path} is missing 'xrays' column.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx >= len(self.df):
            raise IndexError("Index out of range")

        try:
            img_path = self.df.iloc[idx]['xrays']
            image = Image.open(img_path).convert("L")
            image = self.img_transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Returning random tensor.")
            return torch.randn((1, IMG_SIZE, IMG_SIZE)), torch.zeros((1, IMG_SIZE, IMG_SIZE))

        mask_path = None
        if 'masks' in self.df.columns:
            mask_path = self.df.iloc[idx]['masks']

        if pd.isna(mask_path):
            mask = torch.zeros((1, IMG_SIZE, IMG_SIZE))
        else:
            try:
                mask_str_path = str(mask_path)
                mask = Image.open(mask_str_path).convert("L")
                mask = self.mask_transform(mask)
                mask = (mask > 0.5).float()
            except Exception as e:
                print(f"Warning: Could not load mask {mask_str_path}: {e}. Returning zero mask.")
                mask = torch.zeros((1, IMG_SIZE, IMG_SIZE))

        return image, mask