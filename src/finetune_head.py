#!/usr/bin/env python3
"""
finetune_head.py — Fine-tune only the YOLO detection head on a new COCO dataset.
"""

import os
import json
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Import the ComputeLoss class from your utils
import sys
sys.path.append('..')  # Adjust path as needed to import from utils
from utils.util import ComputeLoss

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
        
        # Format targets for ComputeLoss: normalized center coordinates (x_center, y_center, w, h)
        img_width, img_height = im_info["width"], im_info["height"]
        if len(boxes) > 0:
            # Convert to center format and normalize
            x_center = ((boxes[:, 0] + boxes[:, 2]) / 2) / img_width
            y_center = ((boxes[:, 1] + boxes[:, 3]) / 2) / img_height
            width = (boxes[:, 2] - boxes[:, 0]) / img_width
            height = (boxes[:, 3] - boxes[:, 1]) / img_height
            boxes_normalized = torch.stack([x_center, y_center, width, height], dim=1)
        else:
            boxes_normalized = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        # Create target in the format expected by ComputeLoss
        target = {
            'box': boxes_normalized,
            'cls': labels.unsqueeze(1) if len(labels) > 0 else torch.zeros((0, 1), dtype=torch.long),
            'idx': torch.full((len(boxes_normalized), 1), idx) if len(boxes_normalized) > 0 else torch.zeros((0, 1), dtype=torch.long)
        }
        
        if self.transform:
            img = self.transform(img)
        return img, target

def collate_fn(batch):
    """Custom collate function to handle variable numbers of targets"""
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, 0)
    
    # Combine all targets into a single dictionary
    combined_targets = {
        'box': torch.cat([t['box'] for t in targets], 0),
        'cls': torch.cat([t['cls'] for t in targets], 0),
        'idx': torch.cat([t['idx'] for t in targets], 0)
    }
    
    return imgs, combined_targets

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
    # First, let's check what modules exist in the model
    print("Model attributes:", [attr for attr in dir(model) if not attr.startswith('_')])
    
    # Freeze backbone if it exists
    if hasattr(model, 'backbone'):
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("Frozen backbone")
    
    # Freeze neck/FPN if it exists  
    if hasattr(model, 'neck') or hasattr(model, 'fpn'):
        neck_module = getattr(model, 'neck', getattr(model, 'fpn', None))
        if neck_module:
            for p in neck_module.parameters():
                p.requires_grad = False
            print("Frozen neck/FPN")
    
    # Unfreeze head
    if hasattr(model, 'head'):
        for p in model.head.parameters():
            p.requires_grad = True
        print("Unfrozen head")
    else:
        # If no explicit head, try to find detection head modules
        for name, module in model.named_modules():
            if 'head' in name.lower() or 'detect' in name.lower():
                for p in module.parameters():
                    p.requires_grad = True
                print(f"Unfrozen {name}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[info] Trainable params: {sum(p.numel() for p in trainable_params):,}")

    # --- Setup loss function ---
    # Define loss parameters (adjust based on your model)
    loss_params = {
        'box': 7.5,  # box loss gain
        'cls': 0.5,  # cls loss gain  
        'dfl': 1.5,  # dfl loss gain
    }
    compute_loss = ComputeLoss(model, loss_params)

    # --- Dataset ---
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    dataset = CocoDataset(args.imgs, args.ann, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, 
                           num_workers=0, collate_fn=collate_fn)  # Set num_workers=0 to avoid multiprocessing issues

    # --- Optimizer ---
    optimizer = torch.optim.SGD(trainable_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # --- Training loop ---
    print(f"[info] Starting fine-tuning for {args.epochs} epochs on {len(dataset)} images...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_loss_box = 0.0
        running_loss_cls = 0.0
        running_loss_dfl = 0.0

        for imgs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs = imgs.to(device)
            # Move each target component to device
            device_targets = {k: v.to(device) for k, v in targets.items()}
            
            # Forward pass - get model outputs
            outputs = model(imgs)
            
            # Compute loss using ComputeLoss class
            loss_box, loss_cls, loss_dfl = compute_loss(outputs, device_targets)
            loss = loss_box + loss_cls + loss_dfl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_loss_box += loss_box.item()
            running_loss_cls += loss_cls.item()
            running_loss_dfl += loss_dfl.item()

        avg_loss = running_loss / len(dataloader)
        avg_loss_box = running_loss_box / len(dataloader)
        avg_loss_cls = running_loss_cls / len(dataloader)
        avg_loss_dfl = running_loss_dfl / len(dataloader)
        
        print(f"[epoch {epoch+1}] total_loss: {avg_loss:.4f}, box: {avg_loss_box:.4f}, "
              f"cls: {avg_loss_cls:.4f}, dfl: {avg_loss_dfl:.4f}")

        # save checkpoint every epoch
        save_path = f"finetuned_head_epoch{epoch+1}.pt"
        torch.save({"model": model}, save_path)
        print(f"[saved] {save_path}")


if __name__ == "__main__":
    main()