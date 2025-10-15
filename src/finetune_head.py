#!/usr/bin/env python3
"""
finetune_head.py — Fine-tune only the YOLO detection head on a new COCO dataset.

Example:
  python finetune_head.py \
    --weights internal_assets/weights/best.pt \
    --ann COCO/annotations/instances_train.json \
    --imgs COCO/images/train \
    --epochs 10 --batch 8 --lr 1e-3 --device cuda
"""

import os
import json
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ====== dataset loader ======
class CocoDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(ann_file, "r") as f:
            coco = json.load(f)
        self.images = {im["id"]: im for im in coco["images"]}
        self.anns = coco["annotations"]
        self.cats = {cat["id"]: cat["name"] for cat in coco["categories"]}
        self.grouped = {}
        for ann in self.anns:
            self.grouped.setdefault(ann["image_id"], []).append(ann)
        self.ids = list(self.images.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        im_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, im_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        anns = self.grouped.get(img_id, [])
        boxes = []
        labels = []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            img = self.transform(img)
        return img, target


# ====== main ======
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True, help="path to YOLO checkpoint (best.pt)")
    p.add_argument("--ann", type=str, required=True, help="path to COCO annotations json")
    p.add_argument("--imgs", type=str, required=True, help="path to images dir")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Load pretrained model ---
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    if "model" not in ckpt:
        raise RuntimeError(f"{args.weights} does not contain 'model' key.")
    model = ckpt["model"]
    model = model.to(device).float()
    model.train()

    # --- Freeze everything except head ---
    if hasattr(model, "net"):
        for p in model.net.parameters(): p.requires_grad = False
    if hasattr(model, "fpn"):
        for p in model.fpn.parameters(): p.requires_grad = False
    if hasattr(model, "backbone"):
        for p in model.backbone.parameters(): p.requires_grad = False

    for p in model.head.parameters(): p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[info] Trainable params: {sum(p.numel() for p in trainable_params):,}")

    # --- Dataset ---
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    dataset = CocoDataset(args.imgs, args.ann, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=2)

    # --- Optimizer ---
    optimizer = torch.optim.SGD(trainable_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # --- Training loop ---
    print(f"[info] Starting fine-tuning for {args.epochs} epochs on {len(dataset)} images...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for imgs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs = imgs.to(device)
            # your model’s forward() should accept imgs, targets
            loss = model(imgs, targets)
            if isinstance(loss, dict):
                loss = sum(loss.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"[epoch {epoch+1}] avg loss = {avg_loss:.4f}")

        # save checkpoint every epoch
        save_path = f"finetuned_head_epoch{epoch+1}.pt"
        torch.save({"model": model}, save_path)
        print(f"[saved] {save_path}")


if __name__ == "__main__":
    main()