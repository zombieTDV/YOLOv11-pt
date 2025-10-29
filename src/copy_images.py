#!/usr/bin/env python3
"""
copy_images.py
Copy all .jpg/.jpeg files listed in a text file from a source directory into a destination directory.

Example:
  python copy_images.py --list COCO/train.txt --src COCO/images/train --dst COCO/images/new_train
  python copy_images.py --list COCO/train.txt --src COCO/images/train --dst COCO/images/new_train --dry-run
"""

import argparse
from pathlib import Path
import shutil
import sys

def main():
    p = argparse.ArgumentParser(description="Copy jpg files listed in a text file from src dir to dst dir.")
    p.add_argument("--list", required=True, help="Path to the text file listing images (one per line).")
    p.add_argument("--src", required=True, help="Source directory containing the images (e.g. COCO/images/train).")
    p.add_argument("--dst", required=True, help="Destination directory to copy images into (will be created if missing).")
    p.add_argument("--dry-run", action="store_true", help="Show what would be copied, don't actually copy.")
    p.add_argument("--move", action="store_true", help="Move files instead of copying.")
    args = p.parse_args()

    list_path = Path(args.list)
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not list_path.is_file():
        print(f"Error: list file not found: {list_path}", file=sys.stderr)
        sys.exit(2)
    if not src_dir.is_dir():
        print(f"Error: source directory not found: {src_dir}", file=sys.stderr)
        sys.exit(2)

    dst_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    copied = 0
    missing = 0
    skipped = 0

    with list_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or line == "...":
                continue

            # only handle jpg/jpeg (case-insensitive)
            if not line.lower().endswith((".jpg", ".jpeg")):
                skipped += 1
                continue

            # If the list contains paths, take the basename
            filename = Path(line).name
            src_file = src_dir / filename
            dst_file = dst_dir / filename

            total += 1
            if not src_file.exists():
                print(f"[MISSING] {src_file}")
                missing += 1
                continue

            if args.dry_run:
                print(f"[DRY RUN] would {'move' if args.move else 'copy'}: {src_file} -> {dst_file}")
                copied += 1  # count as would-copy
                continue

            try:
                if args.move:
                    shutil.move(str(src_file), str(dst_file))
                else:
                    # copy2 preserves metadata
                    shutil.copy2(str(src_file), str(dst_file))
                print(f"[OK] {src_file.name}")
                copied += 1
            except Exception as e:
                print(f"[ERROR] {src_file} -> {dst_file}: {e}", file=sys.stderr)

    print("\nSummary:")
    print(f"  total listed (jpg/jpeg): {total}")
    print(f"  copied/moved: {copied}")
    print(f"  missing in source: {missing}")
    print(f"  skipped (non-jpg lines): {skipped}")

if __name__ == "__main__":
    main()
