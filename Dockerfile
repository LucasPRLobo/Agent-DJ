FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    rubberband-cli \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for frontend build
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir fastapi uvicorn websockets python-multipart qrcode[pil]

# Frontend build
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN cd frontend && npm ci

COPY frontend/ ./frontend/
RUN cd frontend && npm run build

# Copy remaining files
COPY models/ ./models/
COPY samples/ ./samples/

# Expose ports
EXPOSE 8000 5173

# Default command: run backend (serves API + static frontend)
ENV AGENT_DJ_DB=/app/tracks.db
ENV AGENT_DJ_AUDIO_DIR=/app/samples

CMD ["uvicorn", "agent_dj.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
