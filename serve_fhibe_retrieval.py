"""
Text-to-image retrieval server — multi-dataset edition.

Loads every dataset declared in datasets.yaml at startup.  The UI shows a
tab-switcher when more than one dataset is available.

Endpoints:
  GET /datasets                        list available datasets
  GET /filters?dataset=<id>            filter catalog for a dataset
  GET /search?dataset=<id>&q=...       ranked search with pagination
  GET /image?dataset=<id>&path=...     serve original image
  GET /thumbnails/<id>/<path>          static thumbnails (StaticFiles)

Usage:
  export PHIBE_EMB_DIR=/path/to/phibe_embeddings
  export PHIBE_DATASET_ROOT=/path/to/phibe_dataset
  uvicorn serve_fhibe_retrieval:app --host 0.0.0.0 --port 8000 --workers 1
"""

import csv
import os
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import open_clip
import torch
import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

csv.field_size_limit(sys.maxsize)

MAX_K     = 100
DEFAULT_K = 30

# ----------------------------------------------------------------------------
# CLIP model cache — one model loaded per unique (model, pretrained) pair
# ----------------------------------------------------------------------------

_clip_models: dict[tuple[str, str], tuple] = {}


def _get_clip(model_name: str, pretrained: str) -> tuple:
    key = (model_name, pretrained)
    if key not in _clip_models:
        print(f"[startup] loading CLIP {model_name}/{pretrained}")
        m, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        m = m.eval()
        m.visual = None  # free vision tower — text-only at serve time
        tok = open_clip.get_tokenizer(model_name)
        _clip_models[key] = (m, tok)
        print(f"[startup] {model_name}/{pretrained} ready")
    return _clip_models[key]


# ----------------------------------------------------------------------------
# Per-dataset runtime state
# ----------------------------------------------------------------------------

@dataclass
class DatasetState:
    id: str
    label: str
    clip_key: tuple[str, str]
    embeddings: np.ndarray          # (N, D) float32
    paths: list[str]
    path_to_idx: dict[str, int]
    metadata: dict[str, dict]       # path -> metadata dict
    filter_values: dict[str, list[str]]   # field -> list aligned with paths
    filter_options: dict[str, list[dict]] # field -> [{value, label}, ...]
    filter_fields: list[dict]
    metadata_columns: list[str]
    dataset_root: Path
    emb_dir: Path


def _age_to_bucket(raw: str, buckets: list) -> str:
    try:
        a = int(float(raw))
    except (ValueError, TypeError):
        return ""
    for label, lo, hi in buckets:
        if lo <= a < hi:
            return label
    return ""


def _clean_label(value: str, strip: bool) -> str:
    if not value or not strip:
        return value
    parts = value.split(". ", 1)
    if len(parts) == 2 and parts[0].strip().isdigit():
        return parts[1]
    return value


def _load_dataset(ds_id: str, cfg: dict) -> DatasetState:
    # --- paths ---
    emb_dir      = Path(cfg["emb_dir"])
    dataset_root = Path(cfg["dataset_root"])

    csv_raw = cfg.get("csv") or ""
    if not csv_raw:
        raise RuntimeError(f"No 'csv' path in datasets.yaml for '{ds_id}'")
    csv_path = Path(csv_raw)
    if not csv_path.is_absolute():
        csv_path = dataset_root / csv_path

    clip_model     = cfg.get("clip_model",     "ViT-B-32")
    clip_pretrained = cfg.get("clip_pretrained", "openai")
    clip_key       = (clip_model, clip_pretrained)
    strip          = cfg.get("strip_numeric_prefix", False)
    csv_path_col   = cfg.get("csv_path_col", "filepath")
    path_template  = cfg.get("csv_path_template")   # e.g. "images/{ImageID}.jpg"
    filter_fields  = cfg["filter_fields"]
    meta_columns   = cfg["metadata_columns"]
    age_buckets    = [tuple(b) for b in cfg.get("age_buckets", [])]

    # --- embeddings ---
    emb_file = emb_dir / "embeddings.npz"
    print(f"[{ds_id}] loading embeddings from {emb_file}")
    d = np.load(emb_file, allow_pickle=True)
    embeddings  = d["embeddings"].astype(np.float32)
    paths       = [str(p) for p in d["paths"]]
    path_to_idx = {p: i for i, p in enumerate(paths)}
    print(f"[{ds_id}] {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")

    # preload CLIP (shared across datasets with the same key)
    _get_clip(clip_model, clip_pretrained)

    # --- metadata + filter values ---
    print(f"[{ds_id}] loading metadata from {csv_path}")
    metadata: dict[str, dict] = {}
    filter_values: dict[str, list[str]] = {f["field"]: [""] * len(paths) for f in filter_fields}

    csv_rows: dict[str, dict] = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            fp = path_template.format(**row) if path_template else row[csv_path_col]
            metadata[fp] = {k: row.get(k, "") for k in meta_columns}
            csv_rows[fp] = row

    for i, p in enumerate(paths):
        row = csv_rows.get(p)
        if not row:
            continue
        for fdef in filter_fields:
            field = fdef["field"]
            if fdef.get("derived_type") == "age_bucket":
                src = fdef.get("derived_from", "age")
                filter_values[field][i] = _age_to_bucket(row.get(src, ""), age_buckets)
            elif fdef.get("derived_type") == "max_vote":
                # Pick the label whose annotator vote count is highest.
                # derived_prefix + suffix (str key) = column name; value is vote count.
                prefix    = fdef.get("derived_prefix", "")
                cols_map  = fdef.get("derived_columns", {})
                best_label, best_val = "", -1
                for suffix, label in cols_map.items():
                    try:
                        val = int(float(row.get(prefix + str(suffix), 0) or 0))
                    except (ValueError, TypeError):
                        val = 0
                    if val > best_val:
                        best_val, best_label = val, label
                filter_values[field][i] = best_label if best_val > 0 else ""
            else:
                filter_values[field][i] = row.get(field, "") or ""

    # --- filter options (sorted display list per field) ---
    filter_options: dict[str, list[dict]] = {}
    for fdef in filter_fields:
        field = fdef["field"]
        vals  = list({v for v in filter_values[field] if v})

        def _sort_key(v: str, _s=strip) -> tuple:
            parts = v.split(". ", 1)
            if _s and len(parts) == 2 and parts[0].strip().isdigit():
                return (0, int(parts[0]))
            return (1, v)

        vals.sort(key=_sort_key)
        filter_options[field] = [{"value": v, "label": _clean_label(v, strip)} for v in vals]
        print(f"[{ds_id}] filter '{field}': {len(vals)} distinct values")

    print(f"[{ds_id}] metadata for {len(metadata)} rows")
    return DatasetState(
        id=ds_id, label=cfg.get("label", ds_id.upper()), clip_key=clip_key,
        embeddings=embeddings, paths=paths, path_to_idx=path_to_idx,
        metadata=metadata, filter_values=filter_values, filter_options=filter_options,
        filter_fields=filter_fields, metadata_columns=meta_columns,
        dataset_root=dataset_root, emb_dir=emb_dir,
    )


# ----------------------------------------------------------------------------
# Load all datasets declared in datasets.yaml
# ----------------------------------------------------------------------------

_CONFIG_PATH = Path(os.environ.get("DATASETS_CONFIG", "./datasets.yaml"))
with open(_CONFIG_PATH) as _f:
    _all_config = yaml.safe_load(_f)

DATASETS: dict[str, DatasetState] = {}
for _ds_id, _ds_cfg in _all_config["datasets"].items():
    try:
        DATASETS[_ds_id] = _load_dataset(_ds_id, _ds_cfg)
    except Exception as _e:
        print(f"[warn] skipping dataset '{_ds_id}': {_e}")

if not DATASETS:
    raise RuntimeError("No datasets loaded successfully — check datasets.yaml and env vars")

DEFAULT_DATASET_ID = next(iter(DATASETS))
print(f"[startup] datasets ready: {list(DATASETS)}  default={DEFAULT_DATASET_ID!r}")


# ----------------------------------------------------------------------------
# Search helpers
# ----------------------------------------------------------------------------

def encode_text(query: str, clip_key: tuple[str, str]) -> np.ndarray:
    """Return L2-normalized (D,) float32 text embedding."""
    model, tokenizer = _get_clip(*clip_key)
    with torch.inference_mode():
        tok = tokenizer([query])
        vec = model.encode_text(tok).float()
        vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.squeeze(0).numpy()


@lru_cache(maxsize=512)
def _filter_mask(dataset_id: str, filters_key: tuple) -> Optional[np.ndarray]:
    """(N,) boolean mask for filters_key, or None if no filters. Within-field OR, cross-field AND."""
    if not filters_key:
        return None
    ds = DATASETS[dataset_id]
    mask = np.ones(len(ds.paths), dtype=bool)
    for field, allowed in filters_key:
        if not allowed:
            continue
        allowed_set = set(allowed)
        fv = ds.filter_values[field]
        mask &= np.fromiter((v in allowed_set for v in fv), count=len(ds.paths), dtype=bool)
    return mask


@lru_cache(maxsize=512)
def _rank(dataset_id: str, query: str, filters_key: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_indices, sorted_scores). LRU-cached per (dataset, query, filters)."""
    t0 = time.time()
    ds   = DATASETS[dataset_id]
    mask = _filter_mask(dataset_id, filters_key)

    if query:
        qvec = encode_text(query, ds.clip_key)
        sims = ds.embeddings @ qvec
        if mask is not None:
            sims  = np.where(mask, sims, -np.inf)
            order = np.argsort(-sims)[:int(mask.sum())]
        else:
            order = np.argsort(-sims)
        sorted_scores = sims[order]
    else:
        order         = np.flatnonzero(mask) if mask is not None else np.arange(len(ds.paths))
        sorted_scores = np.zeros(len(order), dtype=np.float32)

    order = order.astype(np.int32)
    print(f"[rank] {dataset_id} q={query!r} filters={len(filters_key)} -> {len(order)} in {(time.time()-t0)*1e3:.1f}ms")
    return order, sorted_scores


def search_page(dataset_id: str, query: str, filters_key: tuple, offset: int, limit: int) -> tuple[list[dict], int]:
    ds = DATASETS[dataset_id]
    order, scores = _rank(dataset_id, query, filters_key)
    total  = len(order)
    offset = max(0, offset)
    limit  = max(0, min(limit, total - offset))
    page   = order[offset:offset + limit]
    return (
        [{"path": ds.paths[int(i)], "score": float(s), "rank": offset + k}
         for k, (i, s) in enumerate(zip(page, scores[offset:offset + limit]))],
        total,
    )


# ----------------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="text-to-image search")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

# Mount each dataset's thumbnail directory at /thumbnails/<dataset_id>/
for _ds_id, _ds in DATASETS.items():
    _thumb_dir = _ds.emb_dir / "thumbnails"
    if _thumb_dir.is_dir():
        app.mount(f"/thumbnails/{_ds_id}", StaticFiles(directory=str(_thumb_dir)), name=f"thumbs_{_ds_id}")
        print(f"[startup] /thumbnails/{_ds_id} -> {_thumb_dir}")


def _get_dataset(dataset_id: str) -> DatasetState:
    ds = DATASETS.get(dataset_id)
    if ds is None:
        raise HTTPException(404, f"unknown dataset '{dataset_id}'; available: {list(DATASETS)}")
    return ds


def _parse_filters(raw: list[str], ds: DatasetState) -> tuple:
    valid = {f["field"] for f in ds.filter_fields}
    by_field: dict[str, set[str]] = {}
    for item in raw:
        if ":" not in item:
            continue
        field, value = item.split(":", 1)
        if field not in valid:
            continue
        by_field.setdefault(field, set()).add(value)
    return tuple(sorted((f, tuple(sorted(vs))) for f, vs in by_field.items()))


# --- endpoints ---------------------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True, "datasets": {ds_id: len(ds.paths) for ds_id, ds in DATASETS.items()}}


@app.get("/datasets")
def datasets_endpoint():
    return JSONResponse({
        "datasets": [
            {"id": ds_id, "label": ds.label, "n_embeddings": len(ds.paths)}
            for ds_id, ds in DATASETS.items()
        ]
    })


@app.get("/filters")
def filters_endpoint(dataset: str = DEFAULT_DATASET_ID):
    ds = _get_dataset(dataset)
    return JSONResponse({
        "fields": [
            {"field": f["field"], "label": f["label"], "options": ds.filter_options[f["field"]]}
            for f in ds.filter_fields
        ]
    })


@app.get("/search")
@limiter.limit("60/minute")
def search_endpoint(
    request: Request,
    dataset: str = DEFAULT_DATASET_ID,
    q: str = "",
    offset: int = 0,
    limit: int = DEFAULT_K,
    filter: list[str] = Query(default_factory=list),
):
    ds = _get_dataset(dataset)
    q  = q.strip()
    filters_key = _parse_filters(filter, ds)
    if not q and not filters_key:
        raise HTTPException(400, "provide a query, a filter, or both")
    if len(q) > 500:
        raise HTTPException(400, "query too long")
    if limit < 1 or limit > MAX_K:
        raise HTTPException(400, f"limit must be in [1, {MAX_K}]")
    if offset < 0:
        raise HTTPException(400, "offset must be >= 0")

    results, total = search_page(dataset, q, filters_key, offset, limit)
    for r in results:
        p = r["path"]
        r["thumb_url"]  = f"/thumbnails/{dataset}/" + str(Path(p).with_suffix(".jpg"))
        r["image_url"]  = f"/image?dataset={dataset}&path=" + p
        r["metadata"]   = ds.metadata.get(p, {})

    return JSONResponse({
        "query": q, "dataset": dataset,
        "filters": [{"field": f, "values": list(v)} for f, v in filters_key],
        "offset": offset, "limit": limit, "total": total,
        "results": results, "has_more": offset + len(results) < total,
    })


@app.get("/image")
@limiter.limit("60/minute")
def full_image(request: Request, dataset: str, path: str):
    ds   = _get_dataset(dataset)
    if path not in ds.path_to_idx:
        raise HTTPException(404, "unknown image")
    full = (ds.dataset_root / path).resolve()
    try:
        full.relative_to(ds.dataset_root.resolve())
    except ValueError:
        raise HTTPException(403, "path escape")
    if not full.is_file():
        raise HTTPException(404, "file missing on disk")
    return FileResponse(full)


# ----------------------------------------------------------------------------
# Single-page UI
# ----------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>text-to-image search</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }
  h1 { font-size: 1.3rem; margin-bottom: .75rem; }
  #dataset-tabs { display: flex; gap: .4rem; margin-bottom: 1rem; flex-wrap: wrap; }
  .tab-btn { padding: .35rem .85rem; font-size: .9rem; border: 1px solid #888; background: transparent; color: inherit; border-radius: 4px; cursor: pointer; }
  .tab-btn.active { background: #2563eb; color: #fff; border-color: #2563eb; }
  form { display: flex; gap: .5rem; margin-bottom: 1.5rem; }
  input[type=text] { flex: 1; padding: .6rem .8rem; font-size: 1rem; border: 1px solid #888; border-radius: 4px; }
  button { padding: .6rem 1rem; font-size: 1rem; cursor: pointer; }
  #status { color: #888; font-size: .9rem; margin-bottom: 1rem; min-height: 1.2em; }
  #filters { display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: 1rem; }
  .filter-btn { position: relative; padding: .35rem .7rem; font-size: .85rem; border: 1px solid #888; background: transparent; color: inherit; border-radius: 999px; cursor: pointer; }
  .filter-btn.active { background: #2563eb; color: #fff; border-color: #2563eb; }
  .filter-btn .count { margin-left: .3rem; opacity: .8; font-size: .75rem; }
  .filter-pop { position: absolute; top: 100%; left: 0; margin-top: .3rem; background: #1a1a1a; color: #eee; border: 1px solid #444; border-radius: 6px; padding: .5rem; min-width: 220px; max-height: 320px; overflow-y: auto; z-index: 5; display: none; text-align: left; }
  .filter-pop.open { display: block; }
  .filter-pop label { display: block; padding: .2rem .3rem; cursor: pointer; font-size: .85rem; white-space: nowrap; }
  .filter-pop label:hover { background: #333; }
  .filter-pop input[type=checkbox] { margin-right: .5rem; }
  .swatch { display: inline-block; width: 14px; height: 14px; border-radius: 3px; border: 1px solid rgba(128,128,128,.5); vertical-align: middle; margin-right: .4rem; }
  #clear-filters { padding: .35rem .7rem; font-size: .85rem; background: transparent; color: #888; border: 1px dashed #888; border-radius: 999px; cursor: pointer; }
  #grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: .75rem; }
  .card { cursor: pointer; position: relative; }
  .card img { width: 100%; aspect-ratio: 3/4; object-fit: cover; border-radius: 4px; display: block; background: #222; }
  .card .score { position: absolute; top: 4px; right: 4px; background: rgba(0,0,0,.7); color: #fff; padding: 1px 6px; border-radius: 3px; font-size: .75rem; }
  #modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.85); z-index: 10; padding: 2rem; overflow: auto; }
  #modal.open { display: flex; gap: 2rem; justify-content: center; align-items: flex-start; flex-wrap: wrap; }
  #modal img { max-width: min(70vw, 800px); max-height: 90vh; object-fit: contain; border-radius: 4px; }
  #meta { background: #111; color: #eee; padding: 1rem 1.5rem; border-radius: 4px; min-width: 280px; max-width: 400px; font-size: .85rem; line-height: 1.5; }
  #meta table { border-collapse: collapse; width: 100%; }
  #meta td { padding: 2px 8px 2px 0; vertical-align: top; }
  #meta td:first-child { color: #999; white-space: nowrap; }
  #close { position: fixed; top: 1rem; right: 1.5rem; color: #fff; font-size: 2rem; cursor: pointer; background: none; border: none; }
</style>
</head>
<body>
<h1>text-to-image search</h1>
<div id="dataset-tabs"></div>
<form id="f">
  <input id="q" type="text" placeholder="describe what you're looking for (or leave empty and use filters only)...">
  <button type="submit">Search</button>
</form>
<div id="filters"></div>
<div id="status"></div>
<div id="grid"></div>
<div id="sentinel" style="height:40px;margin:1rem 0;text-align:center;color:#888;font-size:.85rem;"></div>

<div id="modal" onclick="if(event.target.id==='modal')closeModal()">
  <button id="close" onclick="closeModal()">×</button>
  <img id="modal-img" alt="">
  <div id="meta"></div>
</div>

<script>
const PAGE_SIZE = 30;

const grid       = document.getElementById('grid');
const statusEl   = document.getElementById('status');
const sentinel   = document.getElementById('sentinel');
const modal      = document.getElementById('modal');
const modalImg   = document.getElementById('modal-img');
const metaDiv    = document.getElementById('meta');
const filtersBar = document.getElementById('filters');
const tabsEl     = document.getElementById('dataset-tabs');

let currentDataset = null;
let filterCatalog  = [];
const selected     = new Map();

let state = { query:'', offset:0, total:0, hasMore:false, loading:false, searchT0:0 };

// ---- dataset tabs ----------------------------------------------------------

function renderDatasetTabs(datasets) {
  if (datasets.length <= 1) { tabsEl.style.display = 'none'; return; }
  tabsEl.innerHTML = '';
  for (const ds of datasets) {
    const btn = document.createElement('button');
    btn.className = 'tab-btn';
    btn.dataset.dataset = ds.id;
    btn.textContent = ds.label;
    btn.addEventListener('click', () => switchDataset(ds.id));
    tabsEl.appendChild(btn);
  }
}

async function switchDataset(dsId) {
  currentDataset = dsId;
  for (const btn of document.querySelectorAll('.tab-btn'))
    btn.classList.toggle('active', btn.dataset.dataset === dsId);

  // reset UI state
  selected.clear();
  grid.innerHTML = '';
  statusEl.textContent = '';
  sentinel.textContent = '';
  state = { query:'', offset:0, total:0, hasMore:false, loading:false, searchT0:0 };
  document.getElementById('q').value = '';

  try {
    const res  = await fetch(`/filters?dataset=${encodeURIComponent(dsId)}`);
    const data = await res.json();
    filterCatalog = data.fields;
  } catch(e) { console.error('failed to load filters:', e); filterCatalog = []; }
  renderFilterBar();
  document.getElementById('q').focus();
}

// ---- filter bar ------------------------------------------------------------

function buildFilterParams() {
  const parts = [];
  for (const [field, vals] of selected)
    for (const v of vals) parts.push(`filter=${encodeURIComponent(field+':'+v)}`);
  return parts.join('&');
}

function renderFilterBar() {
  filtersBar.innerHTML = '';
  for (const fdef of filterCatalog) {
    const wrap = document.createElement('div');
    wrap.style.position = 'relative';
    const sel   = selected.get(fdef.field);
    const count = sel ? sel.size : 0;
    const btn   = document.createElement('button');
    btn.type      = 'button';
    btn.className = 'filter-btn' + (count > 0 ? ' active' : '');
    btn.innerHTML = fdef.label + (count > 0 ? `<span class="count">· ${count}</span>` : '');
    const pop = document.createElement('div');
    pop.className = 'filter-pop';
    for (const opt of fdef.options) {
      const lbl = document.createElement('label');
      const cb  = document.createElement('input');
      cb.type    = 'checkbox';
      cb.checked = !!(sel && sel.has(opt.value));
      cb.addEventListener('change', () => {
        if (!selected.has(fdef.field)) selected.set(fdef.field, new Set());
        const s = selected.get(fdef.field);
        cb.checked ? s.add(opt.value) : s.delete(opt.value);
        if (s.size === 0) selected.delete(fdef.field);
        triggerSearch();
        renderFilterBar();
      });
      lbl.appendChild(cb);
      const sw = renderSwatch(fdef.field, opt.value);
      if (sw) lbl.insertAdjacentHTML('beforeend', sw);
      lbl.appendChild(document.createTextNode(' ' + opt.label));
      pop.appendChild(lbl);
    }
    btn.addEventListener('click', e => {
      e.stopPropagation();
      for (const el of document.querySelectorAll('.filter-pop.open'))
        if (el !== pop) el.classList.remove('open');
      pop.classList.toggle('open');
    });
    wrap.appendChild(btn);
    wrap.appendChild(pop);
    filtersBar.appendChild(wrap);
  }
  if (selected.size > 0) {
    const clear = document.createElement('button');
    clear.id        = 'clear-filters';
    clear.type      = 'button';
    clear.textContent = 'Clear filters';
    clear.addEventListener('click', () => { selected.clear(); renderFilterBar(); triggerSearch(); });
    filtersBar.appendChild(clear);
  }
}

document.addEventListener('click', e => {
  if (!e.target.closest('.filter-pop') && !e.target.closest('.filter-btn'))
    for (const el of document.querySelectorAll('.filter-pop.open')) el.classList.remove('open');
});

// ---- search / pagination ---------------------------------------------------

async function fetchPage() {
  if (state.loading || !state.hasMore || !currentDataset) return;
  state.loading = true;
  sentinel.textContent = 'loading...';
  try {
    const qs = [
      `dataset=${encodeURIComponent(currentDataset)}`,
      `q=${encodeURIComponent(state.query)}`,
      `offset=${state.offset}`,
      `limit=${PAGE_SIZE}`,
      buildFilterParams(),
    ].filter(Boolean).join('&');
    const res = await fetch('/search?' + qs);
    if (res.status === 400) {
      sentinel.textContent = '';
      state.loading = false;
      state.hasMore = false;
      statusEl.textContent = 'enter a query or pick at least one filter';
      return;
    }
    if (res.status === 429) { sentinel.textContent = 'rate limited, slow down'; state.loading = false; return; }
    if (!res.ok)            { sentinel.textContent = 'error: ' + res.status;     state.loading = false; return; }
    const data = await res.json();

    for (const r of data.results) {
      const card = document.createElement('div');
      card.className = 'card';
      const scoreLabel = state.query ? r.score.toFixed(3) : '';
      card.innerHTML = `<img loading="lazy" src="${r.thumb_url}" alt="">` +
                       (scoreLabel ? `<span class="score">${scoreLabel}</span>` : '');
      card.onclick = () => openModal(r);
      grid.appendChild(card);
    }

    state.offset += data.results.length;
    state.total   = data.total;
    state.hasMore = data.has_more;

    if (state.offset === data.results.length) {
      const dt    = (performance.now() - state.searchT0).toFixed(0);
      const label = state.query ? 'matching results' : 'results';
      statusEl.textContent = `${data.total} ${label} (${dt}ms). scroll for more.`;
    } else {
      statusEl.textContent = `showing ${state.offset} of ${state.total}`;
    }
    sentinel.textContent = state.hasMore ? '' : `— end (${state.total} results) —`;
  } catch(err) {
    sentinel.textContent = 'error: ' + err.message;
  } finally {
    state.loading = false;
  }
}

function triggerSearch() {
  const q = document.getElementById('q').value.trim();
  if (!q && selected.size === 0) {
    grid.innerHTML = '';
    state.hasMore  = false;
    statusEl.textContent = '';
    sentinel.textContent = '';
    return;
  }
  state = { query:q, offset:0, total:0, hasMore:true, loading:false, searchT0:performance.now() };
  grid.innerHTML       = '';
  statusEl.textContent = 'searching...';
  sentinel.textContent = '';
  fetchPage();
}

document.getElementById('f').addEventListener('submit', e => { e.preventDefault(); triggerSearch(); });

const io = new IntersectionObserver(
  entries => { for (const e of entries) if (e.isIntersecting) fetchPage(); },
  { rootMargin: '400px' }
);
io.observe(sentinel);

// ---- modal -----------------------------------------------------------------

function openModal(r) {
  modalImg.src = r.image_url;
  const rows = Object.entries(r.metadata)
    .filter(([, v]) => v !== '' && v != null)
    .map(([k, v]) => {
      const sw = renderSwatch(k, String(v));
      return `<tr><td>${k}</td><td>${sw}${escapeHtml(String(v))}</td></tr>`;
    }).join('');
  metaDiv.innerHTML = `<table>${rows}</table>`;
  modal.classList.add('open');
}
function closeModal() { modal.classList.remove('open'); modalImg.src = ''; }
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]);
}
function renderSwatch(field, value) {
  if (!/skin_color/.test(field)) return '';
  const m = /\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]/.exec(value || '');
  if (!m) return '';
  return `<span class="swatch" style="background:rgb(${m[1]},${m[2]},${m[3]})"></span>`;
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

// ---- bootstrap -------------------------------------------------------------

(async () => {
  try {
    const res  = await fetch('/datasets');
    const data = await res.json();
    renderDatasetTabs(data.datasets);
    if (data.datasets.length > 0) await switchDataset(data.datasets[0].id);
  } catch(e) { console.error('failed to load datasets:', e); }
})();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML
