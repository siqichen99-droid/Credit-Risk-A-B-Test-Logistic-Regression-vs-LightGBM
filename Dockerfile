# ── Stage 1: Base image ──────────────────────────────────────────────────────
# We use Python 3.11 slim — full Python but without unnecessary OS packages.
# "slim" keeps the image size small (~150MB vs ~900MB for the full image).
FROM python:3.11-slim

# ── Stage 2: System setup ─────────────────────────────────────────────────────
# Set the working directory inside the container.
# All subsequent commands run from this path.
WORKDIR /app

# Prevent Python from writing .pyc files (keeps container clean)
# and force stdout/stderr to be unbuffered (logs appear immediately)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 3: Install Python dependencies ─────────────────────────────────────
# Copy requirements first — Docker caches this layer separately.
# If only your code changes (not requirements), Docker reuses the cached
# pip install layer, making rebuilds much faster.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 4: Copy application files ──────────────────────────────────────────
# Copy only what the API needs to run — not notebooks, data, or dev files
COPY main.py .
COPY Models/ ./models/
COPY ui.html .
COPY Results/feature_cols.txt ./results/feature_cols.txt

# ── Stage 5: Expose and run ───────────────────────────────────────────────────
# Tell Docker this container listens on port 8000
EXPOSE 8000

# Health check — Docker will ping /health every 30s to confirm the API is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the FastAPI server when the container launches
# --host 0.0.0.0 makes it accessible outside the container (not just localhost)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
