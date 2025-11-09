#!/usr/bin/env python3
"""
finetune_change_nc.py — Fine-tune YOLO with a DIFFERENT number of classes (nc).
Loads backbone + neck from pretrained weights, reinitializes head with new nc.
"""

import os
import json
import torch
import argparse
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from utils.util import ComputeLoss, non_max_suppression

class CocoDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(ann_file, "r") as f:
            coco = json.load(f)
        self.images = {im["id"]: im for im in coco["images"]}
        self.ids = [im["id"] for im in coco["images"]]
        self.anns = coco["annotations"]
        self.cats = {cat["id"]: cat["name"] for cat in coco["categories"]}
        self.grouped = {}
        for ann in self.anns:
            self.grouped.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        im_id = self.ids[idx]
        im_info = self.images[im_id]
        img_path = os.path.join(self.img_dir, im_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        anns = self.grouped.get(im_id, [])
        boxes = []
        labels = []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"])

        boxes = torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0,4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long) if len(labels) > 0 else torch.zeros((0,), dtype=torch.long)

        img_width, img_height = im_info["width"], im_info["height"]
        if boxes.shape[0] > 0:
            x_center = ((boxes[:, 0] + boxes[:, 2]) / 2) / img_width
            y_center = ((boxes[:, 1] + boxes[:, 3]) / 2) / img_height
            width = (boxes[:, 2] - boxes[:, 0]) / img_width
            height = (boxes[:, 3] - boxes[:, 1]) / img_height
            boxes_normalized = torch.stack([x_center, y_center, width, height], dim=1)
        else:
            boxes_normalized = torch.zeros((0, 4), dtype=torch.float32)

        target = {
            'box': boxes_normalized,
            'cls': labels.unsqueeze(1) if labels.numel() > 0 else torch.zeros((0,1), dtype=torch.long),
            'idx': torch.full((boxes_normalized.shape[0],1), idx, dtype=torch.long),
            'img_id': torch.full((boxes_normalized.shape[0],1), im_id, dtype=torch.long)
        }

        if self.transform:
            img = self.transform(img)
        return img, target

def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, 0)

    boxes_list, cls_list, idx_list, imgid_list = [], [], [], []
    for batch_i, t in enumerate(targets):
        if t['box'].shape[0] > 0:
            boxes_list.append(t['box'])
            cls_list.append(t['cls'])
            idx_list.append(torch.full((t['box'].shape[0], 1), batch_i, dtype=torch.long))
            if 'img_id' in t:
                imgid_list.append(t['img_id'])
            else:
                imgid_list.append(torch.full((t['box'].shape[0], 1), batch_i, dtype=torch.long))

    if boxes_list:
        combined_targets = {
            'box': torch.cat(boxes_list, 0),
            'cls': torch.cat(cls_list, 0),
            'idx': torch.cat(idx_list, 0),
            'img_id': torch.cat(imgid_list, 0)
        }
    else:
        combined_targets = {
            'box': torch.zeros((0, 4), dtype=torch.float32),
            'cls': torch.zeros((0, 1), dtype=torch.long),
            'idx': torch.zeros((0, 1), dtype=torch.long),
            'img_id': torch.zeros((0, 1), dtype=torch.long)
        }

    return imgs, combined_targets


def load_model_with_new_nc(weights_path, new_nc, device):
    """
    Load pretrained model and replace head with new number of classes.
    
    Strategy:
    1. Load the full pretrained model
    2. Extract backbone and neck (keep their weights)
    3. Create a new head with new_nc
    4. Reconstruct the model
    """
    print(f"[info] Loading pretrained weights from {weights_path}")
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    if "model" not in ckpt:
        raise RuntimeError("checkpoint missing 'model' key")
    
    old_model = ckpt["model"]
    old_nc = old_model.head.nc if hasattr(old_model.head, 'nc') else None
    
    print(f"[info] Old model nc: {old_nc}, New nc: {new_nc}")
    
    if old_nc == new_nc:
        print("[warning] nc unchanged, you can use regular fine-tuning")
        return old_model.to(device).float()
    
    # Get the filters from the old head (for architecture compatibility)
    old_filters = []
    for box_module in old_model.head.box:
        # Get input channels from first conv in the sequential
        old_filters.append(box_module[0].conv.in_channels)
    
    print(f"[info] Detected filters for head: {old_filters}")
    
    # Import your Head class (adjust import as needed)
    from models.yolo import Head  # Adjust this import to match your project structure
    
    # Create new head with new_nc
    new_head = Head(nc=new_nc, filters=tuple(old_filters))
    
    # Copy stride information if available
    if hasattr(old_model.head, 'stride'):
        new_head.stride = old_model.head.stride.clone()
    
    # Initialize biases for new head
    if hasattr(new_head, 'initialize_biases'):
        new_head.initialize_biases()
    
    # Replace the head in the model
    old_model.head = new_head
    
    print(f"[info] Successfully replaced head with nc={new_nc}")
    print(f"[info] New head parameters: {sum(p.numel() for p in new_head.parameters()):,}")
    
    return old_model.to(device).float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True, help="Path to pretrained .pt file")
    p.add_argument("--ann", type=str, required=True, help="COCO annotations JSON")
    p.add_argument("--imgs", type=str, required=True, help="Images directory")
    p.add_argument("--new_nc", type=int, required=True, help="New number of classes")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone during training")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model with modified head
    model = load_model_with_new_nc(args.weights, args.new_nc, device)

    # Configure what to train
    # Strategy 1: Train everything (default)
    # Strategy 2: Freeze backbone, train neck + head (use --freeze_backbone flag)
    
    if args.freeze_backbone:
        print("[info] Freezing backbone")
        if hasattr(model, 'backbone'):
            for p in model.backbone.parameters():
                p.requires_grad = False
    
    # Always train neck and head
    if hasattr(model, 'neck') or hasattr(model, 'fpn'):
        neck_module = getattr(model, 'neck', getattr(model, 'fpn', None))
        if neck_module:
            for p in neck_module.parameters():
                p.requires_grad = True
    
    if hasattr(model, 'head'):
        for p in model.head.parameters():
            p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[info] Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"[info] Total params: {sum(p.numel() for p in model.parameters()):,}")

    # Update ComputeLoss with new nc
    compute_loss = ComputeLoss(model, {'box':7.5, 'cls':0.5, 'dfl':1.5})

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    full_dataset = CocoDataset(args.imgs, args.ann, transform)

    n = len(full_dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    gen = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=gen)

    print(f"[info] Dataset size: total={n}, train={len(train_dataset)}, val={len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

    optimizer = torch.optim.SGD(trainable_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    # Learning rate scheduler (optional but recommended for new head)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_batches = 0

        print(f"[info] Starting epoch {epoch+1}/{args.epochs}, lr={optimizer.param_groups[0]['lr']:.6f}")
        for imgs, targets in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{args.epochs}"):
            imgs = imgs.to(device)
            device_targets = {k: v.to(device) for k, v in targets.items()}

            outputs = model(imgs)

            try:
                loss_box, loss_cls, loss_dfl = compute_loss(outputs, device_targets)
            except Exception as e:
                print("ERROR in ComputeLoss during TRAIN:")
                for k, v in device_targets.items():
                    print(f" target {k}: shape {tuple(v.shape)} dtype {v.dtype}")
                raise

            loss = loss_box + loss_cls + loss_dfl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_batches += 1

        avg_train_loss = running_loss / train_batches if train_batches > 0 else 0.0
        train_losses.append(avg_train_loss)

        # Validation
        model_was_training = model.training
        val_running = 0.0
        val_batches = 0

        with torch.no_grad():
            model.train()  # Keep in train mode for loss computation
            for imgs, targets in tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{args.epochs}"):
                imgs = imgs.to(device)
                device_targets = {k: v.to(device) for k, v in targets.items()}

                outputs = model(imgs)

                try:
                    loss_box, loss_cls, loss_dfl = compute_loss(outputs, device_targets)
                except Exception as e:
                    print("ERROR in ComputeLoss during VAL:")
                    for k, v in device_targets.items():
                        print(f" target {k}: shape {tuple(v.shape)} dtype {v.dtype}")
                    raise

                loss = loss_box + loss_cls + loss_dfl
                val_running += loss.item()
                val_batches += 1

            if not model_was_training:
                model.eval()

        avg_val_loss = val_running / val_batches if val_batches > 0 else 0.0
        val_losses.append(avg_val_loss)

        print(f"[epoch {epoch+1}] train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f}")

        # Step scheduler
        scheduler.step()

        # Save checkpoints
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            os.makedirs("internal_assets/weights", exist_ok=True)
            torch.save({"model": model}, f"internal_assets/weights/finetuned_nc{args.new_nc}_epoch{epoch+1}.pt")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({"model": model}, f"internal_assets/weights/finetuned_nc{args.new_nc}_best.pt")
            print(f"[saved best] finetuned_nc{args.new_nc}_best.pt (val_loss {best_val_loss:.4f})")

    # Plot
    epochs = list(range(1, args.epochs + 1))
    plt.figure()
    plt.plot(epochs, train_losses, label='train_loss')
    plt.plot(epochs, val_losses, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Train vs Val Loss (nc={args.new_nc})')
    plt.legend()
    plt.grid(True)
    out_fig = f"train_val_loss_nc{args.new_nc}.png"
    plt.savefig(out_fig, bbox_inches='tight')
    plt.close()
    print(f"[saved] Loss plot: {out_fig}")


if __name__ == "__main__":
    main()