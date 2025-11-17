import os
import torch
from torch.utils.data import DataLoader
from args import get_args
from dataset import BoneDataset
from trainer import Trainer


def main():
    args = get_args()

    device = "cpu"

    base_dir = '.'
    img_dir = os.path.join(base_dir, 'xrays')
    mask_dir = os.path.join(base_dir, 'masks')

    train_csv_path = os.path.join(args.csv_dir, 'train_set.csv')
    val_csv_path = os.path.join(args.csv_dir, 'val_set.csv')

    # print(f"train: {train_csv_path}")
    # print(f"val: {val_csv_path}")
    # print(f"img: {img_dir}")
    # print(f"mask: {mask_dir}")

    try:
        train_dataset = BoneDataset(csv_path=train_csv_path, img_dir=img_dir, mask_dir=mask_dir)
        val_dataset = BoneDataset(csv_path=val_csv_path, img_dir=img_dir, mask_dir=mask_dir)
    except Exception as e:
        print(f"Dataset: {e}")
        return

    num_workers = max(0, os.cpu_count() // 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    trainer = Trainer(
        args=args,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader
    )

    print(f"training，{args.epochs} epochs...")
    trainer.run()

    print("done")


if __name__ == "__main__":
    main()