#!/usr/bin/env python3
"""
convert_to_coco.py

Converts a simple annotation format (CSV, YOLO txts, or Pascal VOC xmls) to COCO detection format,
copies images into a COCO-style folder structure, and writes train2017.txt / val2017.txt lists.

Usage examples:
    python convert_to_coco.py --images-dir /path/to/images \
        --out-dir ../Dataset/COCO --format csv --ann /path/to/annotations.csv --val-split 0.1 \
        --names utils/args.yaml

    python convert_to_coco.py --images-dir /path/to/images \
        --out-dir ../Dataset/COCO --format yolo --ann /path/to/labels_folder --val-split 0.2
"""

import os
import json
import argparse
import shutil
import random
import csv
from PIL import Image
import xml.etree.ElementTree as ET
import yaml
from glob import glob

def load_names_from_yaml(yaml_path):
    """Read 'names' mapping from utils/args.yaml if present. Return list index->name."""
    if not yaml_path or not os.path.isfile(yaml_path):
        return None
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    raw = cfg.get('names', {})
    # normalize: keys may be strings '0': 'person'
    maxidx = max(int(k) for k in raw.keys()) if raw else -1
    names = [None] * (maxidx + 1)
    for k, v in raw.items():
        idx = int(k)
        names[idx] = str(v)
    return names

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def write_coco_json(images, annotations, categories, out_path):
    coco = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(out_path, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f"[write] Wrote {out_path} with {len(images)} images and {len(annotations)} annotations.")

def bbox_xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return [float(x1), float(y1), float(w), float(h)]

def process_csv(csv_path, images_dir, name2id):
    """
    CSV format: filename,xmin,ymin,xmax,ymax,class
    returns dict: {basename: [ (xmin,ymin,xmax,ymax,category_id), ... ], ...}
    """
    ann = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) < 6:
                continue
            fn = os.path.basename(row[0])
            xmin, ymin, xmax, ymax = map(float, row[1:5])
            class_field = row[5]
            # determine category id
            try:
                cat_id = int(class_field)
            except Exception:
                if class_field in name2id:
                    cat_id = name2id[class_field]
                else:
                    raise KeyError(f"Unknown class name '{class_field}' not found in names mapping.")
            ann.setdefault(fn, []).append((xmin, ymin, xmax, ymax, cat_id))
    return ann

def process_voc(xml_folder, images_dir, name2id):
    ann = {}
    files = glob(os.path.join(xml_folder, "*.xml"))
    for p in files:
        tree = ET.parse(p)
        root = tree.getroot()
        fn = root.find('filename').text
        for obj in root.findall('object'):
            name = obj.find('name').text
            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            ymin = float(xmlbox.find('ymin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymax = float(xmlbox.find('ymax').text)
            if name in name2id:
                cid = name2id[name]
            else:
                try:
                    cid = int(name)
                except:
                    raise KeyError(f"Unknown class name '{name}' in VOC file {p}")
            ann.setdefault(fn, []).append((xmin, ymin, xmax, ymax, cid))
    return ann

def process_yolo(labels_folder, images_dir, name_list):
    """
    expects labels folder with .txt files named like image.jpg -> image.txt lines: class x_center_rel y_center_rel w_rel h_rel
    name_list: list index->name or None
    """
    ann = {}
    # find images in images_dir (common extensions)
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for e in exts:
        image_files += glob(os.path.join(images_dir, e))
    for img_path in image_files:
        base = os.path.basename(img_path)
        txt = os.path.splitext(img_path)[0] + ".txt"
        # prefer labels folder location (labels_folder/<base_no_ext>.txt)
        txt_name = os.path.join(labels_folder, os.path.splitext(base)[0] + ".txt")
        if not os.path.isfile(txt_name):
            # fallback to same folder
            if os.path.isfile(txt):
                txt_name = txt
            else:
                continue
        w,h = Image.open(img_path).size
        with open(txt_name, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_idx = int(parts[0])
                cx_r, cy_r, w_r, h_r = map(float, parts[1:5])
                cx = cx_r * w
                cy = cy_r * h
                bw = w_r * w
                bh = h_r * h
                x1 = cx - bw/2
                y1 = cy - bh/2
                x2 = x1 + bw
                y2 = y1 + bh
                ann.setdefault(base, []).append((x1, y1, x2, y2, cls_idx))
    return ann

def build_coco_from_annmap(ann_map, images_src_dir, out_dir, split='train2017'):
    """
    ann_map: dict basename -> list of (xmin,ymin,xmax,ymax,cat_id)
    copies images into out_dir/images/<split>/ and builds images and annotations arrays
    """
    images_out_dir = os.path.join(out_dir, "images", split)
    ensure_dir(images_out_dir)
    images = []
    annotations = []
    ann_id = 1
    img_id = 1
    for fn, objs in ann_map.items():
        src_img = None
        # search source dir for this image basename
        for ext in ['.jpg','.jpeg','.png','.bmp']:
            cand = os.path.join(images_src_dir, fn)
            if os.path.isfile(cand):
                src_img = cand
                break
            # also try with ext replaced if fn has no ext
            if not os.path.splitext(fn)[1]:
                cand2 = os.path.join(images_src_dir, os.path.splitext(fn)[0] + ext)
                if os.path.isfile(cand2):
                    src_img = cand2
                    break
        if src_img is None:
            # try glob search
            glob_candidates = glob(os.path.join(images_src_dir, os.path.splitext(fn)[0] + ".*"))
            if glob_candidates:
                src_img = glob_candidates[0]
        if src_img is None:
            print(f"[warn] image file for {fn} not found in {images_src_dir}; skipping")
            continue
        # copy to images_out_dir
        dst = os.path.join(images_out_dir, os.path.basename(src_img))
        shutil.copy2(src_img, dst)
        w,h = Image.open(dst).size
        images.append({"id": img_id, "file_name": os.path.basename(dst), "width": w, "height": h})
        for (xmin,ymin,xmax,ymax,cat_id) in objs:
            x = float(max(0, xmin))
            y = float(max(0, ymin))
            x2 = float(max(0, xmax))
            y2 = float(max(0, ymax))
            bw = max(0.0, x2 - x)
            bh = max(0.0, y2 - y)
            bbox = [x, y, bw, bh]
            area = bw * bh
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(cat_id),
                "bbox": bbox,
                "area": area,
                "iscrowd": 0
            })
            ann_id += 1
        img_id += 1
    return images, annotations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, help="Folder with all source images")
    parser.add_argument("--format", required=True, choices=["csv","voc","yolo"], help="annotation input format")
    parser.add_argument("--ann", required=True, help="CSV file / xml folder / yolo labels folder")
    parser.add_argument("--out-dir", required=True, help="Output COCO-style dir (e.g. ../Dataset/COCO)")
    parser.add_argument("--names", default="utils/args.yaml", help="path to args.yaml providing names: mapping (optional)")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    names = load_names_from_yaml(args.names)
    if names:
        print(f"[names] loaded {len(names)} classes from {args.names}")
    else:
        print("[names] no names YAML provided or not found; expecting numeric class IDs in annotations")

    # Build name2id mapping
    name2id = {}
    categories = []
    if names:
        for i, n in enumerate(names):
            if n is None: continue
            name2id[n] = i
            categories.append({"id": i, "name": n})
    else:
        # unknown classes: will build categories dynamically from dataset class ids
        categories = None

    # parse annotations
    if args.format == "csv":
        ann_map = process_csv(args.ann, args.images_dir, name2id)
    elif args.format == "voc":
        ann_map = process_voc(args.ann, args.images_dir, name2id)
    elif args.format == "yolo":
        ann_map = process_yolo(args.ann, args.images_dir, names)
    else:
        raise RuntimeError("unsupported format")

    # turn ann_map filenames into list and split
    all_files = list(ann_map.keys())
    random.seed(args.seed)
    random.shuffle(all_files)
    n_val = int(len(all_files) * args.val_split)
    val_files = set(all_files[:n_val])
    train_files = set(all_files[n_val:])

    # build train and val COCO JSONs
    train_ann_map = {f:ann_map[f] for f in train_files}
    val_ann_map = {f:ann_map[f] for f in val_files}

    images_train, annotations_train = build_coco_from_annmap(train_ann_map, args.images_dir, args.out_dir, split="train2017")
    images_val, annotations_val = build_coco_from_annmap(val_ann_map, args.images_dir, args.out_dir, split="val2017")

    # If categories unknown, derive from annotations
    if categories is None:
        cat_ids = set([a[4] for v in ann_map.values() for a in v])
        categories = [{"id": int(i), "name": str(i)} for i in sorted(cat_ids)]

    # write JSONs
    ann_dir = os.path.join(args.out_dir, "annotations")
    ensure_dir(ann_dir)
    write_coco_json(images_train, annotations_train, categories, os.path.join(ann_dir, "instances_train2017.json"))
    write_coco_json(images_val, annotations_val, categories, os.path.join(ann_dir, "instances_val2017.json"))

    # write train2017.txt and val2017.txt (basenames)
    train_list_path = os.path.join(args.out_dir, "train2017.txt")
    val_list_path = os.path.join(args.out_dir, "val2017.txt")
    with open(train_list_path, "w") as f:
        for im in images_train:
            f.write(im["file_name"] + "\n")
    with open(val_list_path, "w") as f:
        for im in images_val:
            f.write(im["file_name"] + "\n")
    print(f"[lists] wrote {train_list_path} ({len(images_train)}) and {val_list_path} ({len(images_val)})")

if __name__ == "__main__":
    main()
