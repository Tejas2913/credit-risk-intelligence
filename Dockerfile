# ==============================================================================
# Credit Risk Intelligence Platform - Backend Dockerfile
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# No additional system dependencies needed — Python is already available for health checks

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create runtime directories
RUN mkdir -p /app/data /app/artifacts

# Copy application source code and deployment artifacts
COPY src/ /app/src/
COPY artifacts/ /app/artifacts/
COPY sql/ /app/sql/

# Expose FastAPI backend port
EXPOSE 8000

# Container healthcheck using Python's built-in urllib (no curl required)
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

# Start production Uvicorn server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
