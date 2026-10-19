# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Set environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    FLASK_DEBUG=false

WORKDIR /app

# Install curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and pre-computed model artifacts
COPY . .

# Expose internal service port
EXPOSE 5000

# Container healthcheck using /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/health || exit 1

# Launch production WSGI application with Gunicorn
CMD ["sh", "-c", "gunicorn app:app --workers 2 --bind 0.0.0.0:${PORT:-5000} --access-logfile - --error-logfile -"]
