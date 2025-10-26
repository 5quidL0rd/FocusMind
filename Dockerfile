# --------------------------------------------------------------
# 1️⃣  Builder stage – install dependencies, compile wheels
# --------------------------------------------------------------
FROM python:3.11-slim AS builder

# Prevent Python from writing .pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# MediaPipe & OpenCV need a few system libs (glib + libGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create a non‑root user (good practice for Render)
RUN useradd -m appuser

WORKDIR /app

# Copy ONLY the pip requirements first – this maximises Docker cache reuse
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------------
# 2️⃣  Runtime stage – copy source code and run uvicorn
# --------------------------------------------------------------
FROM python:3.11-slim

# System libs needed at runtime (same as build stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Use the same non‑root user we created earlier
RUN useradd -m appuser
USER appuser

WORKDIR /app

# Copy compiled site‑packages from builder (fast, no re‑install)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy the entire source tree – **exclude** the frontend folder (npm files)
COPY . .

# Expose the port Render will set via $PORT (default 10000 on Render)
EXPOSE 10000

# --------------------------------------------------------------
# 3️⃣  Command – start Uvicorn, listening on the env‑provided port
# --------------------------------------------------------------
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "$PORT"]