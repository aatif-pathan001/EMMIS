# Encrypted Multi-Modal Intelligence System
> A Python system that accepts **encrypted text + image** data, runs a full
> NLP + Computer Vision pipeline, and produces a **unified risk score** — all
> behind a Streamlit dashboard and a FastAPI REST API.


## Features

| Feature | Details |
|---|---|
| **Custom Encryption** | XOR (rolling key) → Byte Shift → Byte Scramble → Base64 |
| **NLP Analysis** | DistilBERT sentiment + VADER fallback + 40-keyword risk dict |
| **CV Anomaly Detection** | OpenCV statistical thresholding + contour detection |
| **Risk Scoring** | Random Forest (trained on 2 000 synthetic samples) |
| **Storage** | MongoDB Atlas with automatic in-memory fallback |
| **REST API** | FastAPI with full OpenAPI docs at `/docs` |
| **Dashboard** | Streamlit — 3 pages, Plotly charts, annotated images |
| **Docker** | Multi-service `docker-compose` (API + Dashboard) |


---

## Quick Start
### 1 — Clone / enter the project

```bash
cd EMMIS
```

### 2 — Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
```

### 3 — Install Python dependencies

```bash
# CPU-only PyTorch first (saves ~2 GB versus the full build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# All other dependencies
pip install -e .
```

### 4 — Configure environment

```bash
cp .env.example .env
```

### 5 — Download NLTK data (one-time)

```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
```
---
## Running

### Streamlit Dashboard

The dashboard imports all modules directly — no API server needed.

```bash
streamlit run app.py
```

### FastAPI backend

```bash
uvicorn src.emmis.api.routes:app --reload
```

### Docker Compose

```bash
docker build -t emmis .
docker run --rm --env-file .env -p 8000:8000 -p 8501:8501 emmis
```


---

## API Reference

### `GET /health`
Returns system health and storage backend status.

```json
{
  "status": "healthy",
  "total_records": 12,
  "version": "1.0.0"
}
```

---

### `POST /api/encrypt`

Encrypt plain text (and optional Base64 image).

**Request:**
```json
{
  "text": "Critical failure detected in unit 7",
  "image_base64": "<base64 string>"
}
```

**Response:**
```json
{
  "encrypted_text": "XR8bTq2A...",
  "encrypted_image": "iVBOR...",
  "text_checksum": "d41d8cd9..."
}
```

---

### `POST /api/analyze`

Decrypt and run the full NLP + CV + Risk pipeline.

**Request:**
```json
{
  "encrypted_text": "XR8bTq2A...",
  "encrypted_image": "iVBOR..."
}
```

**Response:**
```json
{
  "record_id": "uuid",
  "timestamp": "2024-01-01T10:00:00Z",
  "decrypted_text": "Critical failure detected in unit 7",
  "nlp_results": {
    "sentiment": { "label": "NEGATIVE", "confidence": 0.97 },
    "risk_keywords": { "found_keywords": ["critical", "failure"], "keyword_count": 2 },
    "nlp_risk_score": 0.82
  },
  "cv_results": {
    "anomaly_detected": true,
    "anomaly_score": 0.68,
    "anomaly_regions": 4
  },
  "risk_assessment": {
    "unified_risk_score": 0.79,
    "risk_level": "HIGH",
    "risk_icon": "🔴",
    "risk_probabilities": { "LOW": 0.03, "MEDIUM": 0.12, "HIGH": 0.85 }
  }
}
```

---

### `GET /api/records?limit=10`

List the most recent analysis records.

---

### `GET /api/records/{record_id}`

Retrieve a single record by UUID.


---

## Encryption Design

The `Cipher` applies **four sequential layers**:

```
Layer 1 — XOR Transform
    Each byte b at position i → b XOR (key XOR (i % 256))
    Rolling XOR ensures identical bytes produce different ciphertext.

Layer 2 — Byte Shift
    Each byte → (byte + shift_value) mod 256
    Shifts all values uniformly through the 0–255 space.

Layer 3 — Byte Scramble
    Bytes are rearranged using a deterministic LCG-seeded permutation.
    Destroys positional patterns.

Layer 4 — Base64 Encode
    Produces a printable, transmissible ASCII string.
```

Decryption applies the exact inverse in reverse order.

---

## Risk Scoring Weights

| Feature | Weight |
|---|---|
| NLP Risk Score | 35% |
| Sentiment Contribution | 25% |
| CV Anomaly Score | 25% |
| Risk Keyword Count | 10% |
| Anomaly Region Count | 5% |


---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Encryption | Custom (XOR + Shift + Scramble + Base64) |
| NLP | HuggingFace Transformers · NLTK VADER |
| CV | OpenCV |
| Risk Model | scikit-learn RandomForestClassifier |
| API | FastAPI · Uvicorn |
| Dashboard | Streamlit · Plotly |
| Database | MongoDB Atlas (pymongo) |
| Containerisation | Docker |
