#!/usr/bin/env python3
"""
update_args_names.py

Scan dataset folders for dataset.yaml files, extract class names and append any new classes
into utils/args.yaml with new numeric indices. Optionally update a top-level `nc` key
in args.yaml (if present).

Usage examples:
    python update_args_names.py --datasets-dir internal_assets/extra_dataset/datasets --args-file utils/args.yaml

    # dry-run (no file write)
    python update_args_names.py --datasets-dir extra_dataset/datasets --args-file utils/args.yaml --dry-run
"""

import argparse
import os
import sys
import yaml
import shutil
from datetime import datetime

# ---------- Utility helpers ----------

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def collect_dataset_yamls(base_dir):
    """Return list of all dataset.yaml files under base_dir"""
    paths = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower() in ("dataset.yaml", "dataset.yml"):
                paths.append(os.path.join(root, f))
    return paths


def extract_names(ds_path):
    """Extract class names list from dataset.yaml"""
    data = load_yaml(ds_path)
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(v).strip() for v in names.values()]
    elif isinstance(names, list):
        return [str(v).strip() for v in names]
    else:
        return []

def extract_names_from_dataset_yaml(path):
    data = load_yaml(path)
    if not data:
        return []
    names = data.get("names")
    if names is None:
        # sometimes dataset.yaml uses "nc" and "names" under other keys; try common alternatives
        return []
    # names can be a list or mapping (0: name)
    if isinstance(names, list):
        return [str(n).strip() for n in names if n is not None]
    elif isinstance(names, dict):
        # the mapping keys might be indices or class strings; collect values if numeric keys,
        # otherwise collect keys
        # decide: if all keys parse to int -> use values, else use values as well (some producers store index: name)
        vals = []
        for k, v in names.items():
            if isinstance(v, str):
                vals.append(v.strip())
            else:
                vals.append(str(v).strip())
        return vals
    else:
        # fallback: convert to string
        return [str(names).strip()]

def build_existing_names_map(args_yaml):
    """
    args_yaml['names'] might be a dict mapping integers to class names (as in the example),
    or a list. Return a dict int->str and a set of class names.
    """
    existing = args_yaml.get("names")
    if existing is None:
        return {}, set()
    if isinstance(existing, list):
        m = {i: str(n).strip() for i, n in enumerate(existing)}
        return m, set(m.values())
    elif isinstance(existing, dict):
        # keys may be string numbers -> convert to int if possible
        m = {}
        for k, v in existing.items():
            try:
                ik = int(k)
            except Exception:
                # fallback: keep increasing keys sequentially if key not int
                ik = None
            if ik is None:
                # assign later in order (collected separately)
                pass
            # store temporarily; we'll normalize indices later
        # Simplest: create an ordered list of values by sorted int keys if possible, else preserve insertion
        try:
            # attempt to order by numeric key
            items = sorted(existing.items(), key=lambda kv: int(kv[0]))
            m = {int(k): str(v).strip() for k, v in items}
        except Exception:
            # fallback: enumerate values to guarantee integer keys
            vals = [str(v).strip() for v in existing.values()]
            m = {i: vals[i] for i in range(len(vals))}
        return m, set(m.values())
    else:
        # unexpected type
        return {}, set()

def apply_new_names_to_args(args_yaml, existing_map, new_names):
    """
    existing_map: dict int->str
    new_names: list[str] of names to add (unique)
    returns updated_args_yaml, added_list
    """
    current_count = max(existing_map.keys()) + 1 if existing_map else 0
    existing_values = set(existing_map.values())
    added = []
    # ensure deterministic order
    for nm in new_names:
        if nm in existing_values:
            continue
        # assign next available index
        idx = current_count
        existing_map[idx] = nm
        existing_values.add(nm)
        added.append((idx, nm))
        current_count += 1

    # write back to args_yaml['names'] as mapping int->name (YAML will show numeric keys)
    # However some users prefer 'names' as mapping 0: name; this achieves that.
    # We'll replace with a dict sorted by index.
    sorted_map = {i: existing_map[i] for i in sorted(existing_map.keys())}
    args_yaml['names'] = sorted_map

    # if args_yaml has top-level 'nc', update it to new total
    if 'nc' in args_yaml:
        args_yaml['nc'] = len(sorted_map)

    return args_yaml, added

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-dir", required=True)
    ap.add_argument("--args-file", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.datasets_dir):
        raise SystemExit(f"Datasets dir not found: {args.datasets_dir}")
    if not os.path.isfile(args.args_file):
        raise SystemExit(f"Args file not found: {args.args_file}")

    # Load args.yaml
    args_yaml = load_yaml(args.args_file)
    existing = args_yaml.get("names", {})

    # Normalize to dict form
    if isinstance(existing, list):
        existing = {i: n for i, n in enumerate(existing)}
    elif not isinstance(existing, dict):
        existing = {}

    existing_names = list(existing.values())
    next_index = len(existing_names)

    # Collect all dataset.yaml files
    ds_paths = collect_dataset_yamls(args.datasets_dir)
    all_new_names = []
    for path in ds_paths:
        ds_names = extract_names(path)
        for name in ds_names:
            if name not in existing_names and name not in all_new_names:
                all_new_names.append(name)

    if not all_new_names:
        print("✅ No new labels found — args.yaml already up-to-date.")
        return

    # Add new names with new indices
    print(f"🆕 Adding {len(all_new_names)} new labels:")
    for name in all_new_names:
        print(f"  {next_index}: {name}")
        existing[next_index] = name
        next_index += 1

    # Update args.yaml
    args_yaml["names"] = existing
    args_yaml["nc"] = len(existing)  # ensure new field is always written

    if args.dry_run:
        print("\n--dry-run enabled: no file changes made --")
        print(yaml.safe_dump(args_yaml, sort_keys=False, allow_unicode=True))
        return

    # Backup old args.yaml
    backup_path = f"{args.args_file}.bak.{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(args.args_file, backup_path)
    print(f"📦 Backup saved to {backup_path}")

    # Save new file
    save_yaml(args_yaml, args.args_file)
    print(f"✅ Updated {args.args_file} successfully! Total classes = {args_yaml['nc']}")


if __name__ == "__main__":
    main()