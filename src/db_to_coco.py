#!/usr/bin/env python3
"""
db_to_coco.py

Minimal converter from MongoDB (or JSON export) docs to COCO detection JSON + train list.

Example:
  python db_to_coco.py --mongo-uri "$env:MONGO_URI" --db face_ml_labeling --collection images --out-dir ./COCO --fs-root /home/user/project

Or:
  python db_to_coco.py --source json --json-file export.json --out-dir ./COCO

"""
import os, json, argparse, shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from PIL import Image

try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None

def ensure_dir(d): 
    Path(d).mkdir(parents=True, exist_ok=True)

def atomic_write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile('w', delete=False, dir=str(path.parent)) as tmp:
        json.dump(data, tmp, indent=2)
        tmp.flush(); os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, str(path))

def xyxy_to_xywh_clamped(x1, y1, x2, y2, img_w, img_h):
    x1c = max(0.0, min(img_w - 1, float(x1)))
    y1c = max(0.0, min(img_h - 1, float(y1)))
    x2c = max(0.0, min(img_w - 1, float(x2)))
    y2c = max(0.0, min(img_h - 1, float(y2)))
    w = max(0.0, x2c - x1c)
    h = max(0.0, y2c - y1c)
    return [x1c, y1c, w, h]

def resolve_file_path(doc, fs_root):
    """
    Try the DB path as-is first. If it doesn't exist and fs_root is given,
    try joining fs_root with the path (strip any leading slash).
    Also allow falling back to fileName if filePath missing.
    """
    file_path = doc.get("filePath") or doc.get("file_path") or doc.get("path") or doc.get("filepath")
    if not file_path:
        file_path = doc.get("fileName") or doc.get("file_name")
        if not file_path:
            return None
        # treat as relative filename
        if fs_root:
            candidate = str(Path(fs_root) / file_path)
        else:
            candidate = file_path
        return candidate

    # try as-is first
    candidate = file_path
    if Path(candidate).exists():
        return candidate

    # if it's absolute but doesn't exist, and fs_root given, try join fs_root + lstrip("/")
    if fs_root:
        candidate2 = str(Path(fs_root) / file_path.lstrip("/"))
        if Path(candidate2).exists():
            return candidate2

    # finally, if it's relative and not found yet, try fs_root + file_path
    if fs_root and not os.path.isabs(file_path):
        candidate3 = str(Path(fs_root) / file_path)
        if Path(candidate3).exists():
            return candidate3

    # not found
    return None

def process_docs(docs, out_dir, fs_root=None, min_conf=0.25):
    out_dir = Path(out_dir)
    images_out = out_dir / "images" / "train"
    ann_dir = out_dir / "annotations"
    ensure_dir(images_out)
    ensure_dir(ann_dir)

    categories_map = {}   # name -> id
    categories_list = []
    next_cat_id = 1  # start from 1

    images = []
    annotations = []
    img_id = 1
    ann_id = 1

    docs = list(docs)
    print(f"[info] processing {len(docs)} documents; min_conf={min_conf}")

    for doc in docs:
        fp = resolve_file_path(doc, fs_root)
        if not fp:
            print(f"[warn] image file not found for doc id {doc.get('_id')} -> skipping")
            continue

        # copy image into COCO images folder
        basename = Path(fp).name
        dst = images_out / basename
        if not dst.exists():
            try:
                shutil.copy2(fp, dst)
            except Exception as e:
                print(f"[warn] failed to copy {fp} -> {e}; skipping")
                continue

        # read image size
        try:
            with Image.open(dst) as im:
                w, h = im.size
        except Exception as e:
            print(f"[warn] cannot open image {dst}: {e}; skipping")
            continue

        images.append({"id": img_id, "file_name": basename, "width": w, "height": h})

        for a in doc.get("annotations", []) or []:
            conf = float(a.get("confidence", 1.0))
            if conf < min_conf:
                continue
            bbox = a.get("bbox") or a.get("box")
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = bbox[:4]
            xywh = xyxy_to_xywh_clamped(x1, y1, x2, y2, w, h)
            if xywh[2] <= 0 or xywh[3] <= 0:
                continue

            label = a.get("label")
            if label is None:
                label = a.get("category_id")
            # numeric label?
            if isinstance(label, (int, float)) or (isinstance(label, str) and label.isdigit()):
                cat_id = int(label)
                if cat_id not in [c['id'] for c in categories_list]:
                    categories_list.append({"id": cat_id, "name": str(cat_id)})
                    categories_map[str(cat_id)] = cat_id
                    next_cat_id = max(next_cat_id, cat_id+1)
            else:
                label_name = str(label)
                if label_name in categories_map:
                    cat_id = categories_map[label_name]
                else:
                    cat_id = next_cat_id
                    categories_map[label_name] = cat_id
                    categories_list.append({"id": cat_id, "name": label_name})
                    next_cat_id += 1

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(cat_id),
                "bbox": [float(x) for x in xywh],
                "area": float(xywh[2] * xywh[3]),
                "iscrowd": 0,
                "score": float(conf)
            })
            ann_id += 1

        img_id += 1

    coco_out = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": sorted(categories_list, key=lambda x: int(x['id']))
    }

    ann_path = ann_dir / "instances_train.json"
    atomic_write_json(ann_path, coco_out)

    with open(out_dir / "train.txt", "w") as f:
        for im in images:
            f.write(im["file_name"] + "\n")

    print(f"[done] Wrote {ann_path} images:{len(images)} anns:{len(annotations)} cats:{len(categories_list)}")
    return ann_path

def main():
    p = argparse.ArgumentParser(description="Minimal MongoDB -> COCO converter")
    p.add_argument("--source", choices=["mongo", "json"], default="mongo")
    p.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    p.add_argument("--db", default="test")
    p.add_argument("--collection", default="images")
    p.add_argument("--json-file", help="If --source json, path to exported JSON list")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-conf", type=float, default=0.25)
    p.add_argument("--fs-root", default=None, help="prefix to prepend to DB filePath fields if needed")
    args = p.parse_args()

    docs = []
    if args.source == "mongo":
        if MongoClient is None:
            raise RuntimeError("pymongo required for mongo source")
        client = MongoClient(args.mongo_uri)
        db = client[args.db]
        coll = db[args.collection]
        cursor = coll.find({})
        for d in cursor:
            docs.append(d)
    else:
        with open(args.json_file, 'r') as f:
            docs = json.load(f)
            if not isinstance(docs, list):
                raise RuntimeError("JSON must be a list of documents")

    ann_path = process_docs(docs, args.out_dir, fs_root=args.fs_root, min_conf=args.min_conf)
    print("COCO json path:", ann_path)

if __name__ == "__main__":
    main()
