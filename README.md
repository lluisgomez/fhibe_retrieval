# fhibe_retrieval

```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

# On the GPU box:
```
pip install -r requirements_embed.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

# On the CPU serving box:
```
pip install -r requirements_serve.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

```
export FHIBE_EMB_DIR=/data/datasets/FHIBE/fhibe_embeddings
export FHIBE_DATASET_ROOT=/data/datasets/FHIBE/fhibe.20250716.u.gT5_rFTA_downsampled_public
export FHIBE_CSV=$FHIBE_DATASET_ROOT/data/processed/fhibe_downsampled/fhibe_downsampled.csv

uvicorn serve_fhibe_retrieval:app --host 0.0.0.0 --port 8000
```
