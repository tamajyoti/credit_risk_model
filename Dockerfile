# ── Stage 1: prepare data ─────────────────────────────────────────────────────
# Runs prepare_data.py to build training_set.csv, scaler.joblib, scaler_params.json
# Uses the raw CSVs from data/ and the pre-trained model.joblib from artifacts/
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/          data/
COPY artifacts/     artifacts/
COPY prepare_data.py .

RUN python prepare_data.py

# ── Stage 2: API ───────────────────────────────────────────────────────────────
# Copies only what the API needs no raw data, no build tools
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY app.py .

# Copy artifacts produced by the builder stage
# includes: model.joblib, scaler.joblib, scaler_params.json, training_set.csv
COPY --from=builder /app/artifacts/ artifacts/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]