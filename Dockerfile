# project/Dockerfile

# Frontend build stage
FROM node:20-slim as frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ .
ARG ENVIRONMENT=production
COPY frontend/.env.${ENVIRONMENT} .env.production
RUN npm run build

# Backend build stage
FROM python:3.12-slim as backend-builder

WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Final stage
FROM python:3.12-slim

# Create non-root user
RUN useradd -m -u 1000 bibleai

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies
COPY --from=backend-builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=backend-builder /usr/local/bin/ /usr/local/bin/

# Copy frontend build
COPY --from=frontend-builder /app/frontend/build /app/frontend/build

# Copy application code
COPY . .

# Set ownership
RUN chown -R bibleai:bibleai /app

# Switch to non-root user
USER bibleai

# Expose ports
EXPOSE 8000
EXPOSE 8050

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FRONTEND_BUILD_PATH=/app/frontend/build

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "app.py"]
