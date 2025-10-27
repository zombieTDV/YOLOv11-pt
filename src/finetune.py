#!/usr/bin/env python3
"""
finetune.py — Fine-tune only the YOLO detection head on a new COCO dataset.
Fixed to call model in train-mode during forward for loss computation (even in validation),
so ComputeLoss receives the same output format as training. Keeps torch.no_grad() for val.
Also 80/20 split, plotting, diagnostics.
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
            # dataset idx is kept for debugging, collate will produce batch-relative idx
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--ann", type=str, required=True)
    p.add_argument("--imgs", type=str, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    if "model" not in ckpt:
        raise RuntimeError("checkpoint missing 'model' key")
    model = ckpt["model"].to(device).float()

    # Freeze/unfreeze same as before
    if hasattr(model, 'backbone'):
        for p in model.backbone.parameters():
            p.requires_grad = True
    if hasattr(model, 'neck') or hasattr(model, 'fpn'):
        neck_module = getattr(model, 'neck', getattr(model, 'fpn', None))
        if neck_module:
            for p in neck_module.parameters():
                p.requires_grad = True
    if hasattr(model, 'head'):
        for p in model.head.parameters():
            p.requires_grad = True
    else:
        for name, module in model.named_modules():
            if 'head' in name.lower() or 'detect' in name.lower():
                for p in module.parameters():
                    p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[info] Trainable params: {sum(p.numel() for p in trainable_params):,}")

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

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # ensure model is in train mode for training forward
        model.train()
        running_loss = 0.0
        train_batches = 0

        print(f"[info] Starting epoch {epoch+1}/{args.epochs}")
        for imgs, targets in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{args.epochs}"):
            imgs = imgs.to(device)
            device_targets = {k: v.to(device) for k, v in targets.items()}

            # forward (train mode)
            outputs = model(imgs)

            # diagnostic print
            # if isinstance(outputs, (list, tuple)):
            #     for i, o in enumerate(outputs):
            #         print(f" TRAIN outputs[{i}] shape: {tuple(o.shape)}  numel={o.numel()}")
            # else:
            #     print(" TRAIN single output shape:", tuple(outputs.shape))

            try:
                loss_box, loss_cls, loss_dfl = compute_loss(outputs, device_targets)
            except Exception as e:
                print("ERROR in ComputeLoss during TRAIN. Diagnostic targets keys/shapes:")
                for k, v in device_targets.items():
                    print(f" TRAIN target {k}: shape {tuple(v.shape)} dtype {v.dtype}")
                raise

            loss = loss_box + loss_cls + loss_dfl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_batches += 1

        avg_train_loss = running_loss / train_batches if train_batches > 0 else 0.0
        train_losses.append(avg_train_loss)

        # --------- Validation: IMPORTANT change ----------
        # Do the forward in train-mode (so model returns the same multi-scale outputs
        # that ComputeLoss expects) but keep torch.no_grad() so we do not compute grads.
        model_was_training = model.training  # store
        val_running = 0.0
        val_batches = 0

        with torch.no_grad():
            # Set model to train mode (only for forward behavior), but inside no_grad so no grads are collected
            model.train()
            for imgs, targets in tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{args.epochs}"):
                imgs = imgs.to(device)
                device_targets = {k: v.to(device) for k, v in targets.items()}

                outputs = model(imgs)

                # # diagnostic print for val
                # if isinstance(outputs, (list, tuple)):
                #     for i, o in enumerate(outputs):
                #         print(f" VAL outputs[{i}] shape: {tuple(o.shape)}  numel={o.numel()}")
                # else:
                #     print(" VAL single output shape:", tuple(outputs.shape))

                # try:
                #     loss_box, loss_cls, loss_dfl = compute_loss(outputs, device_targets)
                # except Exception as e:
                #     print("ERROR in ComputeLoss during VAL. Diagnostic targets keys/shapes:")
                #     for k, v in device_targets.items():
                #         print(f" VAL target {k}: shape {tuple(v.shape)} dtype {v.dtype} min/max:{None if v.numel()==0 else (v.min().item(), v.max().item())}")
                #     raise

                loss = loss_box + loss_cls + loss_dfl
                val_running += loss.item()
                val_batches += 1

            # restore model mode to what it was before validation
            if not model_was_training:
                model.eval()

        avg_val_loss = val_running / val_batches if val_batches > 0 else 0.0
        val_losses.append(avg_val_loss)

        print(f"[epoch {epoch+1}] train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f}")

        # save checkpoints
        if epoch % 10 == 0:
            torch.save({"model": model}, f"finetuned_head_epoch{epoch+1}.pt")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({"model": model}, "finetuned_head_best.pt")
                print(f"[saved best] finetuned_head_best.pt (val_loss {best_val_loss:.4f})")

    # plot train/val loss vs epoch
    epochs = list(range(1, args.epochs + 1))
    plt.figure()
    plt.plot(epochs, train_losses, label='train_loss')
    plt.plot(epochs, val_losses, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Val Loss')
    plt.legend()
    plt.grid(True)
    out_fig = "train_val_loss.png"
    plt.savefig(out_fig, bbox_inches='tight')
    plt.close()
    print(f"[saved] Loss plot: {out_fig}")


if __name__ == "__main__":
    main()
