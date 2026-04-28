"""
Precompute CLIP image embeddings for any dataset declared in datasets.yaml.

Reads image paths from the dataset's CSV manifest, encodes each image with
open_clip, and saves L2-normalised float16 embeddings plus 384px JPEG
thumbnails.

Outputs (under --out-dir, resolved from config if not given):
  embeddings.npz    -- {'embeddings': (N, D) float16, 'paths': (N,) str}
  thumbnails/<relative_path>.jpg  -- mirrors source directory structure

Usage:
  # Full run (paths from env vars declared in datasets.yaml)
  python embed_dataset.py --dataset phibe --device cuda

  # Override paths on the command line
  python embed_dataset.py --dataset phibe \\
      --dataset-root /data/phibe --out-dir /data/phibe_emb

  # Smoke test on 100 images, CPU
  python embed_dataset.py --dataset phibe --limit 100 --device cpu

  # Resume after interruption
  python embed_dataset.py --dataset phibe --device cuda --resume
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)


# ----------------------------------------------------------------------------
# Config helpers
# ----------------------------------------------------------------------------

def _load_cfg(config_path: Path, dataset_id: str) -> dict:
    with open(config_path) as f:
        all_cfg = yaml.safe_load(f)
    available = list(all_cfg.get("datasets", {}))
    if dataset_id not in all_cfg.get("datasets", {}):
        sys.exit(f"Dataset '{dataset_id}' not in {config_path}. Available: {available}")
    return all_cfg["datasets"][dataset_id]


def _resolve_paths(cfg: dict, args) -> tuple[Path, Path, Path]:
    """Return (dataset_root, emb_dir, csv_path).

    Priority: CLI arg > datasets.yaml value.
    """
    dataset_root = Path(args.dataset_root) if args.dataset_root else Path(cfg["dataset_root"])
    emb_dir      = Path(args.out_dir)      if args.out_dir      else Path(cfg["emb_dir"])

    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_raw = cfg.get("csv") or sys.exit("No 'csv' path in datasets.yaml for this dataset")
        csv_path = Path(csv_raw)
        if not csv_path.is_absolute():
            csv_path = dataset_root / csv_path

    return dataset_root, emb_dir, csv_path


# ----------------------------------------------------------------------------
# Manifest loading
# ----------------------------------------------------------------------------

def load_manifest(
    csv_path: Path,
    path_col: str,
    path_template: str | None,
    primary_filter: dict | None,
    all_rows: bool,
) -> list[str]:
    """Return relative image paths from the manifest CSV.

    Path is built from path_template (a str.format pattern over row columns,
    e.g. "images/{ImageID}.jpg") when set, otherwise from path_col directly.
    If primary_filter is set and all_rows is False, only rows matching
    primary_filter['column'] == primary_filter['value'] are included.
    """
    paths = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if not all_rows and primary_filter:
                col, val = primary_filter["column"], primary_filter["value"]
                if row.get(col, "").lower() != val.lower():
                    continue
            paths.append(path_template.format(**row) if path_template else row[path_col])
    return paths


# ----------------------------------------------------------------------------
# PyTorch dataset + collate
# ----------------------------------------------------------------------------

class ImageDataset(Dataset):
    def __init__(
        self,
        rel_paths: list[str],
        root: Path,
        preprocess,
        thumb_dir: Path,
        thumb_size: int = 384,
    ):
        self.rel_paths  = rel_paths
        self.root       = root
        self.preprocess = preprocess
        self.thumb_dir  = thumb_dir
        self.thumb_size = thumb_size

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, idx):
        rel      = self.rel_paths[idx]
        abs_path = self.root / rel
        try:
            img = Image.open(abs_path).convert("RGB")
        except Exception as e:
            return {"rel": rel, "tensor": None, "error": str(e)}

        thumb_path = (self.thumb_dir / rel).with_suffix(".jpg")
        if not thumb_path.exists():
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            thumb = img.copy()
            thumb.thumbnail((self.thumb_size, self.thumb_size), Image.LANCZOS)
            thumb.save(thumb_path, "JPEG", quality=85, optimize=True)

        return {"rel": rel, "tensor": self.preprocess(img), "error": None}


def collate(batch):
    good = [b for b in batch if b["tensor"] is not None]
    bad  = [(b["rel"], b["error"]) for b in batch if b["tensor"] is None]
    if not good:
        return None, [], bad
    return torch.stack([b["tensor"] for b in good]), [b["rel"] for b in good], bad


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Embed images for a dataset defined in datasets.yaml")
    ap.add_argument("--dataset",      required=True,  help="Dataset ID (must match a key in datasets.yaml)")
    ap.add_argument("--config",       type=Path, default=Path("./datasets.yaml"), help="Path to datasets.yaml")
    # Optional path overrides (default: resolved from config + env vars)
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--out-dir",      type=Path, default=None)
    ap.add_argument("--csv",          type=Path, default=None)
    # Model overrides (default: from config)
    ap.add_argument("--model",        default=None, help="CLIP model name (overrides config)")
    ap.add_argument("--pretrained",   default=None, help="CLIP pretrained weights (overrides config)")
    # Embedding options
    ap.add_argument("--batch-size",   type=int,  default=256)
    ap.add_argument("--num-workers",  type=int,  default=8)
    ap.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit",        type=int,  default=None, help="Process first N images only (smoke test)")
    ap.add_argument("--thumb-size",   type=int,  default=384)
    ap.add_argument("--resume",       action="store_true", help="Skip paths already in embeddings.npz")
    ap.add_argument("--all-rows",     action="store_true", help="Include non-primary rows (skip primary_filter)")
    args = ap.parse_args()

    cfg = _load_cfg(args.config, args.dataset)
    dataset_root, emb_dir, csv_path = _resolve_paths(cfg, args)

    model_name    = args.model      or cfg.get("clip_model",      "ViT-B-32")
    pretrained    = args.pretrained or cfg.get("clip_pretrained", "openai")
    path_col      = cfg.get("csv_path_col", "filepath")
    path_template = cfg.get("csv_path_template")   # e.g. "images/{ImageID}.jpg"
    prim_filter   = cfg.get("primary_filter")       # e.g. {"column": "is_primary", "value": "true"}

    emb_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir = emb_dir / "thumbnails"
    thumb_dir.mkdir(exist_ok=True)
    emb_path  = emb_dir / "embeddings.npz"

    print(f"[setup] dataset={args.dataset}  device={args.device}  model={model_name}/{pretrained}")
    print(f"[setup] dataset_root={dataset_root}")
    print(f"[setup] out_dir={emb_dir}")
    print(f"[setup] reading manifest: {csv_path}")

    rel_paths = load_manifest(csv_path, path_col, path_template, prim_filter, args.all_rows)
    print(f"[setup] manifest rows: {len(rel_paths)}")

    # Resume: skip already-embedded paths
    prev_embs, prev_paths = None, None
    done: set[str] = set()
    if args.resume and emb_path.exists():
        prev = np.load(emb_path, allow_pickle=True)
        prev_paths = list(prev["paths"])
        prev_embs  = prev["embeddings"]
        done       = set(prev_paths)
        print(f"[resume] already have {len(done)} embeddings; skipping those")

    rel_paths = [p for p in rel_paths if p not in done]
    if args.limit is not None:
        rel_paths = rel_paths[: args.limit]
    print(f"[setup] to process: {len(rel_paths)}")

    if not rel_paths:
        print("nothing to do.")
        return

    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model      = model.to(args.device).eval()
    embed_dim  = model.visual.output_dim
    print(f"[setup] embed_dim={embed_dim}")

    ds = ImageDataset(rel_paths, dataset_root, preprocess, thumb_dir, args.thumb_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(args.device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    all_embs: list[np.ndarray] = []
    all_rels: list[str]        = []
    errors:   list[tuple]      = []

    t0 = time.time()
    with torch.inference_mode():
        for tensors, rels, bad in tqdm(loader, desc="embedding"):
            errors.extend(bad)
            if tensors is None:
                continue
            tensors = tensors.to(args.device, non_blocking=True)
            if args.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = model.encode_image(tensors)
            else:
                feats = model.encode_image(tensors)
            feats = feats.float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embs.append(feats.cpu().numpy().astype(np.float16))
            all_rels.extend(rels)

    dt = time.time() - t0
    n  = len(all_rels)
    print(f"[done] embedded {n} images in {dt:.1f}s ({n / max(dt, 1e-9):.1f} img/s)")
    if errors:
        print(f"[warn] {len(errors)} images failed to load; first 5:")
        for rel, err in errors[:5]:
            print(f"    {rel}: {err}")

    new_embs = np.concatenate(all_embs, axis=0) if all_embs else np.zeros((0, embed_dim), dtype=np.float16)
    if prev_embs is not None:
        embs  = np.concatenate([prev_embs, new_embs], axis=0)
        paths = prev_paths + all_rels
    else:
        embs  = new_embs
        paths = all_rels

    np.savez(emb_path, embeddings=embs, paths=np.array(paths))
    print(f"[save] {emb_path}  shape={embs.shape}  dtype={embs.dtype}")
    print(f"[save] thumbnails in {thumb_dir}")


if __name__ == "__main__":
    main()
