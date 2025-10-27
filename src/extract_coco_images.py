"""
extract_coco_images.py

Reads src/COCO/annotations/instances_train.json, finds all image filenames listed
under "images" -> "file_name", and copies those files from src/uploads/images/
into src/COCO/extracted_images/.

Works case-insensitively and logs missing / extra files.
"""

import json
from pathlib import Path
import shutil
import sys

# --- CONFIGURE PATHS HERE ---
COCO_JSON = Path("COCO/annotations/instances_train.json")
UPLOADS_DIR = Path("uploads/images")
OUT_DIR = Path("uploads/extracted_images")
# -----------------------------

def main():
    if not COCO_JSON.exists():
        print(f"ERROR: JSON not found at: {COCO_JSON}")
        sys.exit(1)
    if not UPLOADS_DIR.exists():
        print(f"ERROR: uploads images dir not found: {UPLOADS_DIR}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load JSON
    with COCO_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    json_filenames = [img.get("file_name") for img in images if img.get("file_name")]
    json_filenames_set = set(json_filenames)
    if not json_filenames:
        print("Warning: no 'images' -> 'file_name' entries found in JSON.")
        return

    # Build a case-insensitive map for files found in uploads folder
    uploads_files = list(UPLOADS_DIR.iterdir())
    uploads_map = {p.name.lower(): p for p in uploads_files if p.is_file()}

    copied = []
    missing = []
    for fn in json_filenames_set:
        if fn is None:
            continue
        candidate = uploads_map.get(fn.lower())
        if candidate:
            dest = OUT_DIR / candidate.name
            try:
                # copy2 preserves metadata; overwrite if exists
                shutil.copy2(candidate, dest)
                copied.append(candidate.name)
            except Exception as e:
                print(f"Failed to copy {candidate} -> {dest}: {e}")
        else:
            missing.append(fn)

    # Also optionally show which files were present in uploads but not listed in JSON
    uploads_not_listed = [p.name for p in uploads_files if p.is_file() and p.name.lower() not in {n.lower() for n in json_filenames_set}]

    # Report
    print("---- Extraction report ----")
    print(f"Total filenames listed in JSON (unique): {len(json_filenames_set)}")
    print(f"Copied files: {len(copied)}")
    if copied:
        print("  Examples:", ", ".join(copied[:10]) + (", ..." if len(copied) > 10 else ""))
    print(f"Missing on disk (listed in JSON but not found in uploads): {len(missing)}")
    if missing:
        print("  Examples:", ", ".join(missing[:10]) + (", ..." if len(missing) > 10 else ""))
    print(f"Files in uploads/ but NOT listed in JSON: {len(uploads_not_listed)}")
    if uploads_not_listed:
        print("  Examples:", ", ".join(uploads_not_listed[:10]) + (", ..." if len(uploads_not_listed) > 10 else ""))
    print(f"All matched images copied to: {OUT_DIR.resolve()}")
    print("---------------------------")

if __name__ == "__main__":
    main()
