#!/usr/bin/env python3
"""
db_to_coco.py

Minimal converter from MongoDB (or JSON export) docs to COCO detection JSON + train list.

This version will read utils/args.yaml (if present) and use the `names:` mapping there
to determine category IDs so the produced COCO file uses the same numeric ids the model/visualizer expects.
"""
import os, json, argparse, shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from PIL import Image

try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None

# New: yaml import for reading utils/args.yaml
try:
    import yaml
except Exception:
    yaml = None

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

# New helper: try to load utils/args.yaml and return a dict mapping lowercase name -> int id
def load_names_from_args_yaml():
    """
    Look for utils/args.yaml relative to this script and return a mapping name_lower -> id (int).
    If yaml library missing or file not found, returns an empty dict.
    """
    name2id = {}
    try:
        if yaml is None:
            return {}
        script_dir = Path(__file__).resolve().parent
        candidate = script_dir / "utils" / "args.yaml"
        if not candidate.exists():
            # also try relative to current working directory
            candidate = Path("utils") / "args.yaml"
            if not candidate.exists():
                return {}
        with open(candidate, 'r') as f:
            data = yaml.safe_load(f)
        names_block = data.get('names') if isinstance(data, dict) else None
        if names_block is None:
            return {}
        # names_block may be a dict {0: 'person', 1: 'bicycle', ...} or a list.
        if isinstance(names_block, dict):
            for k, v in names_block.items():
                try:
                    kid = int(k)
                except Exception:
                    # if keys are strings that are numeric, still cast; otherwise skip
                    try:
                        kid = int(str(k))
                    except Exception:
                        continue
                name2id[str(v).lower()] = kid
        elif isinstance(names_block, list):
            for i, v in enumerate(names_block):
                name2id[str(v).lower()] = int(i)
    except Exception:
        return {}
    return name2id

def process_docs(docs, out_dir, fs_root=None, min_conf=0.25):
    out_dir = Path(out_dir)
    images_out = out_dir / "images" / "train"
    ann_dir = out_dir / "annotations"
    ensure_dir(images_out)
    ensure_dir(ann_dir)

    # Load names mapping from utils/args.yaml (if present)
    names_map = load_names_from_args_yaml()  # name_lower -> id (int)
    if names_map:
        print(f"[info] Loaded {len(names_map)} names from utils/args.yaml")
    else:
        print("[info] No utils/args.yaml mapping found or yaml not available; falling back to auto ids")

    categories_map = {}   # name (original-case) -> id
    categories_list = []
    # If names_map present, pre-populate categories_list and categories_map so IDs match the model's expectation
    used_ids = set()
    if names_map:
        # create canonical categories entries for all names in names_map
        # keep original name casing from names_map keys? we only have lower-case keys from loader,
        # so store title-cased name for readability while preserving id.
        for name_lower, cid in sorted(names_map.items(), key=lambda x: int(x[1])):
            # preserve the original label using the lowercase key? We'll use the lowercase key as the canonical lookup
            display_name = name_lower  # keep lower-case to match DB lookups in lower-case
            categories_list.append({"id": int(cid), "name": display_name})
            categories_map[display_name] = int(cid)
            used_ids.add(int(cid))

    next_cat_id = (max(used_ids) + 1) if used_ids else 1  # start after pre-populated ids (or 1)

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

            cat_id = None
            # numeric label?
            if isinstance(label, (int, float)) or (isinstance(label, str) and str(label).isdigit()):
                # use numeric label directly (but ensure we don't clash with pre-populated ids)
                cat_id = int(label)
                if cat_id not in used_ids:
                    # preserve numeric label by adding to categories_list
                    categories_list.append({"id": cat_id, "name": str(cat_id)})
                    categories_map[str(cat_id)] = cat_id
                    used_ids.add(cat_id)
                    next_cat_id = max(next_cat_id, cat_id + 1)
            else:
                # treat as string label name; prefer names_map if it contains this name (case-insensitive)
                label_name = str(label).strip()
                label_lower = label_name.lower()
                if label_lower in categories_map:
                    # categories_map stores lower-case keys when pre-populated
                    cat_id = categories_map[label_lower]
                elif label_lower in names_map:
                    # name exists in utils/args.yaml; adopt that id
                    cat_id = int(names_map[label_lower])
                    categories_map[label_lower] = cat_id
                    # also add to categories_list if not already present
                    if cat_id not in used_ids:
                        categories_list.append({"id": cat_id, "name": label_lower})
                        used_ids.add(cat_id)
                    next_cat_id = max(next_cat_id, cat_id + 1)
                else:
                    # unknown name: assign a new id (after existing ones)
                    cat_id = next_cat_id
                    categories_map[label_lower] = cat_id
                    categories_list.append({"id": cat_id, "name": label_lower})
                    used_ids.add(cat_id)
                    next_cat_id += 1

            # ensure we have an integer cat_id
            if cat_id is None:
                continue

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

    # Sort categories by id for readability
    categories_sorted = sorted(categories_list, key=lambda x: int(x['id']))

    coco_out = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories_sorted
    }

    ann_path = ann_dir / "instances_train.json"
    atomic_write_json(ann_path, coco_out)

    with open(out_dir / "train.txt", "w") as f:
        for im in images:
            f.write(im["file_name"] + "\n")

    print(f"[done] Wrote {ann_path} images:{len(images)} anns:{len(annotations)} cats:{len(categories_sorted)}")
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
