"""
Precompute CLIP ViT-B/32 image embeddings for the FHIBE dataset.

Reads filepaths from the dataset CSV (filtering is_primary=True), encodes each
image with open_clip, and saves L2-normalized float16 embeddings plus a parallel
list of relative paths. Also generates 384px JPEG thumbnails in the same pass.

Outputs (under --out-dir):
  embeddings.npz    -- {'embeddings': (N, 512) float16, 'paths': (N,) str}
  thumbnails/<relative_path>.jpg  -- 384px JPEGs, mirrors source dir structure

Usage:
  # Smoke test on 100 images, CPU is fine
  python embed_fhibe.py --limit 100 --device cpu

  # Full run on GPU
  python embed_fhibe.py --device cuda --batch-size 256 --num-workers 8

  # Resume after interruption (skips paths already in embeddings.npz)
  python embed_fhibe.py --device cuda --resume
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import open_clip

# FHIBE CSV has huge polygon fields in some columns
csv.field_size_limit(sys.maxsize)


def load_manifest(csv_path: Path, primary_only: bool = True) -> list[str]:
    """Return relative filepaths from the manifest, optionally filtered to primary."""
    paths = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if primary_only and row["is_primary"].lower() != "true":
                continue
            paths.append(row["filepath"])
    return paths


class ImageDataset(Dataset):
    def __init__(self, rel_paths: list[str], root: Path, preprocess, thumb_dir: Path, thumb_size: int = 384):
        self.rel_paths = rel_paths
        self.root = root
        self.preprocess = preprocess
        self.thumb_dir = thumb_dir
        self.thumb_size = thumb_size

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, idx):
        rel = self.rel_paths[idx]
        abs_path = self.root / rel
        try:
            img = Image.open(abs_path).convert("RGB")
        except Exception as e:
            # Return a sentinel; collate_fn will drop these
            return {"rel": rel, "tensor": None, "error": str(e)}

        # Thumbnail: save as JPEG alongside, mirroring the relative path
        thumb_path = self.thumb_dir / rel
        thumb_path = thumb_path.with_suffix(".jpg")
        if not thumb_path.exists():
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            thumb = img.copy()
            thumb.thumbnail((self.thumb_size, self.thumb_size), Image.LANCZOS)
            thumb.save(thumb_path, "JPEG", quality=85, optimize=True)

        tensor = self.preprocess(img)
        return {"rel": rel, "tensor": tensor, "error": None}


def collate(batch):
    """Drop items that failed to load; return tensors + rels + errors."""
    good = [b for b in batch if b["tensor"] is not None]
    bad = [(b["rel"], b["error"]) for b in batch if b["tensor"] is None]
    if not good:
        return None, [], bad
    tensors = torch.stack([b["tensor"] for b in good])
    rels = [b["rel"] for b in good]
    return tensors, rels, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/datasets/FHIBE/fhibe.20250716.u.gT5_rFTA_downsampled_public"),
        help="Root that contains data/raw/... and data/processed/...",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to manifest CSV (default: <root>/data/processed/fhibe_downsampled/fhibe_downsampled.csv)",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("/data/datasets/FHIBE/fhibe_embeddings"))
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N images (smoke test)")
    ap.add_argument("--thumb-size", type=int, default=384)
    ap.add_argument("--resume", action="store_true", help="Skip paths already in existing embeddings.npz")
    ap.add_argument("--all-rows", action="store_true", help="Include non-primary rows (default: primary only)")
    args = ap.parse_args()

    csv_path = args.csv or (args.dataset_root / "data/processed/fhibe_downsampled/fhibe_downsampled.csv")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir = args.out_dir / "thumbnails"
    thumb_dir.mkdir(exist_ok=True)
    emb_path = args.out_dir / "embeddings.npz"

    print(f"[setup] device={args.device}  model={args.model}/{args.pretrained}")
    print(f"[setup] reading manifest: {csv_path}")
    rel_paths = load_manifest(csv_path, primary_only=not args.all_rows)
    print(f"[setup] manifest rows: {len(rel_paths)}")

    # Resume logic
    done = set()
    prev_embs, prev_paths = None, None
    if args.resume and emb_path.exists():
        prev = np.load(emb_path, allow_pickle=True)
        prev_paths = list(prev["paths"])
        prev_embs = prev["embeddings"]
        done = set(prev_paths)
        print(f"[resume] already have {len(done)} embeddings; skipping those")

    rel_paths = [p for p in rel_paths if p not in done]
    if args.limit is not None:
        rel_paths = rel_paths[: args.limit]
    print(f"[setup] to process: {len(rel_paths)}")

    if not rel_paths:
        print("nothing to do.")
        return

    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model = model.to(args.device).eval()
    embed_dim = model.visual.output_dim
    print(f"[setup] embed_dim={embed_dim}")

    ds = ImageDataset(rel_paths, args.dataset_root, preprocess, thumb_dir, args.thumb_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(args.device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    all_embs: list[np.ndarray] = []
    all_rels: list[str] = []
    errors: list[tuple[str, str]] = []

    t0 = time.time()
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if args.device == "cuda"
        else torch.inference_mode()  # no-op wrapper for CPU; we use inference_mode below anyway
    )

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
    n = len(all_rels)
    print(f"[done] embedded {n} images in {dt:.1f}s ({n/max(dt,1e-9):.1f} img/s)")
    if errors:
        print(f"[warn] {len(errors)} images failed to load; first 5:")
        for rel, err in errors[:5]:
            print(f"    {rel}: {err}")

    # Merge with previous (resume) and save
    new_embs = np.concatenate(all_embs, axis=0) if all_embs else np.zeros((0, embed_dim), dtype=np.float16)
    if prev_embs is not None:
        embs = np.concatenate([prev_embs, new_embs], axis=0)
        paths = prev_paths + all_rels
    else:
        embs = new_embs
        paths = all_rels

    np.savez(emb_path, embeddings=embs, paths=np.array(paths))
    print(f"[save] {emb_path}  embeddings={embs.shape} {embs.dtype}")
    print(f"[save] thumbnails in {thumb_dir}")


if __name__ == "__main__":
    main()
