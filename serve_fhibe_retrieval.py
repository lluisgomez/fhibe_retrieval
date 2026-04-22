"""
FHIBE text-to-image retrieval server.

Single FastAPI app that:
  - loads CLIP ViT-B/32 text encoder + precomputed image embeddings at startup
  - exposes GET /search?q=...&k=20 returning top-K matches with metadata
  - serves thumbnails (fast, ~30KB JPEGs) and full original PNGs
  - renders a minimal HTML page with a search box, grid, and detail modal

Layout expected on disk:
  EMB_DIR/
    embeddings.npz          # {'embeddings': (N,512) float16, 'paths': (N,) str}
    thumbnails/<relpath>.jpg
  DATASET_ROOT/              # original PNGs live here, under the paths in embeddings.npz
  CSV_PATH                   # fhibe_downsampled.csv (for per-image metadata)

Usage:
  pip install fastapi uvicorn[standard] open_clip_torch torch pillow numpy slowapi
  export FHIBE_EMB_DIR=/path/to/fhibe_embeddings
  export FHIBE_DATASET_ROOT=/path/to/fhibe.20250716.u..._public
  export FHIBE_CSV=/path/to/fhibe_downsampled.csv
  uvicorn serve_fhibe:app --host 0.0.0.0 --port 8000 --workers 1
"""

import csv
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import open_clip

csv.field_size_limit(sys.maxsize)

# ----------------------------------------------------------------------------
# Config via env vars (easier to swap paths without editing code)
# ----------------------------------------------------------------------------
EMB_DIR = Path(os.environ.get("FHIBE_EMB_DIR", "./fhibe_embeddings"))
DATASET_ROOT = Path(os.environ.get("FHIBE_DATASET_ROOT", "."))
CSV_PATH = Path(
    os.environ.get(
        "FHIBE_CSV",
        str(DATASET_ROOT / "data/processed/fhibe_downsampled/fhibe_downsampled.csv"),
    )
)
MODEL_NAME = os.environ.get("FHIBE_MODEL", "ViT-B-32")
PRETRAINED = os.environ.get("FHIBE_PRETRAINED", "openai")
MAX_K = 100      # max per-page size (pagination)
DEFAULT_K = 30   # default page size

# Columns to expose in the detail view (subset of 77 — keep it human-readable)
METADATA_COLUMNS = [
    "subject_id", "age", "pronoun", "nationality", "ancestry",
    "apparent_skin_color", "hairstyle", "apparent_hair_type",
    "apparent_hair_color", "facial_hairstyle", "apparent_left_eye_color",
    "apparent_right_eye_color", "scene", "lighting", "weather",
    "action_body_pose", "action_subject_object_interaction",
    "camera_position", "camera_distance", "manufacturer", "model",
    "location_country", "location_region",
]

# Fields exposed as multi-select filters in the UI.
# `derived` means we compute the value from a different column (see build_filter_value).
FILTER_FIELDS = [
    {"field": "nationality",          "label": "Nationality"},
    {"field": "ancestry",             "label": "Ancestry"},
    {"field": "pronoun",              "label": "Pronoun"},
    {"field": "apparent_skin_color",  "label": "Apparent skin color"},
    {"field": "scene",                "label": "Scene"},
    {"field": "action_body_pose",     "label": "Body pose"},
    {"field": "age_bucket",           "label": "Age", "derived": True},
]

AGE_BUCKETS = [
    ("0-12 (child)",       0, 13),
    ("13-17 (teen)",       13, 18),
    ("18-29 (young adult)", 18, 30),
    ("30-49 (adult)",      30, 50),
    ("50-64 (older adult)", 50, 65),
    ("65+ (senior)",       65, 200),
]


def age_to_bucket(raw: str) -> str:
    """Turn '24' into '18-29 (young adult)'. Returns '' if unparseable."""
    try:
        a = int(float(raw))
    except (ValueError, TypeError):
        return ""
    for label, lo, hi in AGE_BUCKETS:
        if lo <= a < hi:
            return label
    return ""


def clean_label(value: str) -> str:
    """Strip 'N. ' or 'NN. ' prefix used in FHIBE categorical values for display."""
    if not value:
        return value
    # e.g. "12. Indoor: Home or hotel" -> "Indoor: Home or hotel"
    parts = value.split(". ", 1)
    if len(parts) == 2 and parts[0].strip().isdigit():
        return parts[1]
    return value

# ----------------------------------------------------------------------------
# Load artifacts at startup
# ----------------------------------------------------------------------------
print(f"[startup] loading embeddings from {EMB_DIR/'embeddings.npz'}")
_d = np.load(EMB_DIR / "embeddings.npz", allow_pickle=True)
EMBEDDINGS: np.ndarray = _d["embeddings"].astype(np.float32)  # (N, 512) for CPU matmul
PATHS: list[str] = [str(p) for p in _d["paths"]]
PATH_TO_IDX: dict[str, int] = {p: i for i, p in enumerate(PATHS)}
print(f"[startup] {EMBEDDINGS.shape[0]} embeddings loaded, dim={EMBEDDINGS.shape[1]}")

print(f"[startup] loading CLIP text encoder: {MODEL_NAME}/{PRETRAINED}")
_model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
_model = _model.eval()
_tokenizer = open_clip.get_tokenizer(MODEL_NAME)
# Free vision tower — we only need text at query time
_model.visual = None
print("[startup] model loaded, vision tower discarded")

print(f"[startup] loading metadata from {CSV_PATH}")
METADATA: dict[str, dict] = {}
# For each filter field, a list aligned with PATHS: FILTER_VALUES[field][i] = value for image i
FILTER_VALUES: dict[str, list[str]] = {f["field"]: [""] * len(PATHS) for f in FILTER_FIELDS}

_csv_rows_by_path: dict[str, dict] = {}
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        fp = row["filepath"]
        METADATA[fp] = {k: row.get(k, "") for k in METADATA_COLUMNS}
        _csv_rows_by_path[fp] = row

# Populate FILTER_VALUES in the order of PATHS so index i lines up with EMBEDDINGS[i]
for i, p in enumerate(PATHS):
    row = _csv_rows_by_path.get(p)
    if not row:
        continue
    for fdef in FILTER_FIELDS:
        field = fdef["field"]
        if fdef.get("derived") and field == "age_bucket":
            FILTER_VALUES[field][i] = age_to_bucket(row.get("age", ""))
        else:
            FILTER_VALUES[field][i] = row.get(field, "") or ""
del _csv_rows_by_path

# Catalog: for each filter field, the sorted list of distinct values + display labels.
# Sent to the UI so it can render the multi-select dropdowns.
FILTER_OPTIONS: dict[str, list[dict]] = {}
for fdef in FILTER_FIELDS:
    field = fdef["field"]
    vals = sorted({v for v in FILTER_VALUES[field] if v})
    # Preserve FHIBE's numeric-prefix ordering: "0. Standing" < "10. ..." when sorted as numbers
    def _sort_key(v: str):
        parts = v.split(". ", 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            return (0, int(parts[0]))
        return (1, v)
    vals.sort(key=_sort_key)
    FILTER_OPTIONS[field] = [{"value": v, "label": clean_label(v)} for v in vals]
    print(f"[startup] filter '{field}': {len(vals)} distinct values")

print(f"[startup] metadata for {len(METADATA)} rows")


def encode_text(query: str) -> np.ndarray:
    """Return L2-normalized (512,) float32 text embedding."""
    with torch.inference_mode():
        tok = _tokenizer([query])
        vec = _model.encode_text(tok).float()
        vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.squeeze(0).numpy()


from functools import lru_cache


def _filter_mask(filters_key: tuple[tuple[str, tuple[str, ...]], ...]) -> Optional[np.ndarray]:
    """Given a normalized filter tuple, return a (N,) boolean mask, or None if no filters.

    filters_key is a tuple of (field, tuple_of_allowed_values). Within a field it's OR,
    across fields it's AND.
    """
    if not filters_key:
        return None
    mask = np.ones(len(PATHS), dtype=bool)
    for field, allowed in filters_key:
        if not allowed:
            continue
        allowed_set = set(allowed)
        field_vals = FILTER_VALUES[field]
        field_mask = np.fromiter((v in allowed_set for v in field_vals), count=len(PATHS), dtype=bool)
        mask &= field_mask
    return mask


@lru_cache(maxsize=256)
def _rank(query: str, filters_key: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_indices, sorted_scores) after applying filters.

    If query is empty, returns filtered indices in original CSV order with zero scores.
    If query is present, ranks by cosine similarity among matching rows only.
    Cached per (query, filters) pair.
    """
    t0 = time.time()
    mask = _filter_mask(filters_key)

    if query:
        qvec = encode_text(query)              # (512,)
        sims = EMBEDDINGS @ qvec               # (N,) float32
        if mask is not None:
            sims = np.where(mask, sims, -np.inf)
            # argsort moves -inf to the end; we trim them off
            order = np.argsort(-sims)
            n_valid = int(mask.sum())
            order = order[:n_valid]
            sorted_scores = sims[order]
        else:
            order = np.argsort(-sims)
            sorted_scores = sims[order]
    else:
        # No query: just return filtered indices in original order, scores = 0
        if mask is None:
            order = np.arange(len(PATHS))
        else:
            order = np.flatnonzero(mask)
        sorted_scores = np.zeros(len(order), dtype=np.float32)

    order = order.astype(np.int32)
    dt = (time.time() - t0) * 1000
    print(f"[rank] q={query!r} filters={len(filters_key)} -> {len(order)} results in {dt:.1f}ms")
    return order, sorted_scores


def search_page(
    query: str,
    filters_key: tuple,
    offset: int,
    limit: int,
) -> tuple[list[dict], int]:
    """Return a slice of ranked/filtered results plus the total count."""
    order, sorted_scores = _rank(query, filters_key)
    total = len(order)
    offset = max(0, offset)
    limit = max(0, min(limit, total - offset))
    page_idx = order[offset : offset + limit]
    page_scores = sorted_scores[offset : offset + limit]
    results = [
        {
            "path": PATHS[int(i)],
            "score": float(s),
            "rank": offset + k,
        }
        for k, (i, s) in enumerate(zip(page_idx, page_scores))
    ]
    return results, total


# ----------------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="FHIBE text-to-image search")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Mount thumbnails as static files — nginx can later short-circuit this path for speed
app.mount(
    "/thumbnails",
    StaticFiles(directory=str(EMB_DIR / "thumbnails")),
    name="thumbnails",
)


@app.get("/health")
def health():
    return {"ok": True, "n_embeddings": len(PATHS)}


@app.get("/filters")
def filters_endpoint():
    """Return the catalog of filter fields and their available values for the UI."""
    return JSONResponse({
        "fields": [
            {
                "field": f["field"],
                "label": f["label"],
                "options": FILTER_OPTIONS[f["field"]],
            }
            for f in FILTER_FIELDS
        ],
    })


def _parse_filters(raw: list[str]) -> tuple:
    """Parse ['nationality:Filipino', 'scene:12. Indoor: ...'] into a sorted tuple suitable as cache key.

    Within a field, multiple values OR together. Across fields, AND.
    """
    valid_fields = {f["field"] for f in FILTER_FIELDS}
    by_field: dict[str, set[str]] = {}
    for item in raw:
        if ":" not in item:
            continue
        field, value = item.split(":", 1)
        if field not in valid_fields:
            continue
        by_field.setdefault(field, set()).add(value)
    # Sort everything so equivalent filter sets produce identical cache keys
    return tuple(sorted(
        (field, tuple(sorted(values)))
        for field, values in by_field.items()
    ))


@app.get("/search")
@limiter.limit("60/minute")
def search_endpoint(
    request: Request,
    q: str = "",
    offset: int = 0,
    limit: int = DEFAULT_K,
    filter: list[str] = Query(default_factory=list),  # noqa: B008 — FastAPI idiom
):
    q = q.strip()
    filters_key = _parse_filters(filter)
    if not q and not filters_key:
        raise HTTPException(400, "provide a query, a filter, or both")
    if len(q) > 500:
        raise HTTPException(400, "query too long")
    if limit < 1 or limit > MAX_K:
        raise HTTPException(400, f"limit must be in [1, {MAX_K}]")
    if offset < 0:
        raise HTTPException(400, "offset must be >= 0")
    results, total = search_page(q, filters_key, offset, limit)
    for r in results:
        p = r["path"]
        r["thumb_url"] = "/thumbnails/" + str(Path(p).with_suffix(".jpg"))
        r["image_url"] = "/image?path=" + p
        r["metadata"] = METADATA.get(p, {})
    return JSONResponse({
        "query": q,
        "filters": [{"field": f, "values": list(v)} for f, v in filters_key],
        "offset": offset,
        "limit": limit,
        "total": total,
        "results": results,
        "has_more": offset + len(results) < total,
    })


@app.get("/image")
@limiter.limit("60/minute")
def full_image(request: Request, path: str):
    """Serve the original PNG. Validate path is one we know about (prevents LFI)."""
    if path not in PATH_TO_IDX:
        raise HTTPException(404, "unknown image")
    full = (DATASET_ROOT / path).resolve()
    # Defense in depth: ensure resolved path is still under DATASET_ROOT
    try:
        full.relative_to(DATASET_ROOT.resolve())
    except ValueError:
        raise HTTPException(403, "path escape")
    if not full.is_file():
        raise HTTPException(404, "file missing on disk")
    return FileResponse(full, media_type="image/png")


# ----------------------------------------------------------------------------
# Minimal single-page UI
# ----------------------------------------------------------------------------
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FHIBE search</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }
  h1 { font-size: 1.3rem; margin-bottom: 1rem; }
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
<h1>FHIBE text-to-image search</h1>
<form id="f">
  <input id="q" type="text" placeholder="describe what you're looking for (or leave empty and use filters only)...">
  <button type="submit">Search</button>
</form>
<div id="filters"></div>
<div id="status"></div>
<div id="grid"></div>
<div id="sentinel" style="height: 40px; margin: 1rem 0; text-align: center; color: #888; font-size: .85rem;"></div>

<div id="modal" onclick="if(event.target.id==='modal')closeModal()">
  <button id="close" onclick="closeModal()">×</button>
  <img id="modal-img" alt="">
  <div id="meta"></div>
</div>

<script>
const PAGE_SIZE = 30;

const grid = document.getElementById('grid');
const statusEl = document.getElementById('status');
const sentinel = document.getElementById('sentinel');
const modal = document.getElementById('modal');
const modalImg = document.getElementById('modal-img');
const metaDiv = document.getElementById('meta');
const filtersBar = document.getElementById('filters');

// Selected filter values: { field: Set<value> }
const selected = new Map();

let filterCatalog = [];  // fetched from /filters

let state = {
  query: '',
  offset: 0,
  total: 0,
  hasMore: false,
  loading: false,
  searchT0: 0,
};

function buildFilterParams() {
  const params = [];
  for (const [field, values] of selected) {
    for (const v of values) params.push(`filter=${encodeURIComponent(field + ':' + v)}`);
  }
  return params.join('&');
}

function renderFilterBar() {
  filtersBar.innerHTML = '';
  for (const fdef of filterCatalog) {
    const wrap = document.createElement('div');
    wrap.style.position = 'relative';
    const sel = selected.get(fdef.field);
    const count = sel ? sel.size : 0;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'filter-btn' + (count > 0 ? ' active' : '');
    btn.innerHTML = `${fdef.label}${count > 0 ? `<span class="count">· ${count}</span>` : ''}`;
    const pop = document.createElement('div');
    pop.className = 'filter-pop';
    for (const opt of fdef.options) {
      const lbl = document.createElement('label');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = !!(sel && sel.has(opt.value));
      cb.addEventListener('change', () => {
        if (!selected.has(fdef.field)) selected.set(fdef.field, new Set());
        const s = selected.get(fdef.field);
        if (cb.checked) s.add(opt.value); else s.delete(opt.value);
        if (s.size === 0) selected.delete(fdef.field);
        // Re-run the current search with new filters
        triggerSearch();
        // Update the pill label
        renderFilterBar();
      });
      lbl.appendChild(cb);
      // For skin-color fields, show a swatch before the label text
      const swatch = renderSwatch(fdef.field, opt.value);
      if (swatch) lbl.insertAdjacentHTML('beforeend', swatch);
      lbl.appendChild(document.createTextNode(' ' + opt.label));
      pop.appendChild(lbl);
    }
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      // Close other open popups
      for (const el of document.querySelectorAll('.filter-pop.open')) {
        if (el !== pop) el.classList.remove('open');
      }
      pop.classList.toggle('open');
    });
    wrap.appendChild(btn);
    wrap.appendChild(pop);
    filtersBar.appendChild(wrap);
  }
  if (selected.size > 0) {
    const clear = document.createElement('button');
    clear.id = 'clear-filters';
    clear.type = 'button';
    clear.textContent = 'Clear filters';
    clear.addEventListener('click', () => {
      selected.clear();
      renderFilterBar();
      triggerSearch();
    });
    filtersBar.appendChild(clear);
  }
}

// Close filter popups when clicking outside
document.addEventListener('click', (e) => {
  if (!e.target.closest('.filter-pop') && !e.target.closest('.filter-btn')) {
    for (const el of document.querySelectorAll('.filter-pop.open')) el.classList.remove('open');
  }
});

async function fetchPage() {
  if (state.loading || !state.hasMore) return;
  state.loading = true;
  sentinel.textContent = 'loading...';
  try {
    const qs = [
      `q=${encodeURIComponent(state.query)}`,
      `offset=${state.offset}`,
      `limit=${PAGE_SIZE}`,
      buildFilterParams(),
    ].filter(Boolean).join('&');
    const res = await fetch('/search?' + qs);
    if (res.status === 400) {
      // Probably no query and no filters
      sentinel.textContent = '';
      state.loading = false;
      state.hasMore = false;
      statusEl.textContent = 'enter a query or pick at least one filter';
      return;
    }
    if (res.status === 429) { sentinel.textContent = 'rate limited, slow down'; state.loading = false; return; }
    if (!res.ok) { sentinel.textContent = 'error: ' + res.status; state.loading = false; return; }
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
    state.total = data.total;
    state.hasMore = data.has_more;

    if (state.offset === data.results.length) {
      const dt = (performance.now() - state.searchT0).toFixed(0);
      const label = state.query ? 'matching results' : 'results';
      statusEl.textContent = `${data.total} ${label} (${dt}ms). scroll for more.`;
    } else {
      statusEl.textContent = `showing ${state.offset} of ${state.total}`;
    }
    sentinel.textContent = state.hasMore ? '' : `— end (${state.total} results) —`;
  } catch (err) {
    sentinel.textContent = 'error: ' + err.message;
  } finally {
    state.loading = false;
  }
}

function triggerSearch() {
  const q = document.getElementById('q').value.trim();
  // Don't issue a request if there's nothing to search for at all
  if (!q && selected.size === 0) {
    grid.innerHTML = '';
    state.hasMore = false;
    statusEl.textContent = '';
    sentinel.textContent = '';
    return;
  }
  state = { query: q, offset: 0, total: 0, hasMore: true, loading: false, searchT0: performance.now() };
  grid.innerHTML = '';
  statusEl.textContent = 'searching...';
  sentinel.textContent = '';
  fetchPage();
}

document.getElementById('f').addEventListener('submit', (e) => {
  e.preventDefault();
  triggerSearch();
});

const io = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) fetchPage();
}, { rootMargin: '400px' });
io.observe(sentinel);

function openModal(r) {
  modalImg.src = r.image_url;
  const rows = Object.entries(r.metadata)
    .filter(([k, v]) => v !== '' && v != null)
    .map(([k, v]) => {
      const swatch = renderSwatch(k, String(v));
      return `<tr><td>${k}</td><td>${swatch}${escapeHtml(String(v))}</td></tr>`;
    })
    .join('');
  metaDiv.innerHTML = `<table>${rows}</table>`;
  modal.classList.add('open');
}
function closeModal() { modal.classList.remove('open'); modalImg.src = ''; }
function escapeHtml(s) { return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]); }

// If `field` is a skin-color field and `value` looks like "N. [R, G, B]",
// return an inline-block color-swatch HTML snippet. Otherwise return ''.
function renderSwatch(field, value) {
  if (!/skin_color/.test(field)) return '';
  const m = /\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]/.exec(value || '');
  if (!m) return '';
  const [_, r, g, b] = m;
  return `<span class="swatch" style="background: rgb(${r},${g},${b})"></span>`;
}

document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

// Bootstrap: fetch filter catalog before user does anything
(async () => {
  try {
    const res = await fetch('/filters');
    const data = await res.json();
    filterCatalog = data.fields;
    renderFilterBar();
  } catch (err) {
    console.error('failed to load filters:', err);
  }
  document.getElementById('q').focus();
})();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML
