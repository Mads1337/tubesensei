FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home --shell /bin/bash appuser

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn==21.2.0

# Copy application code
COPY tubesensei/ ./tubesensei/
COPY templates/ ./templates/
COPY static/ ./static/

# Create directories that the app may need at runtime
RUN mkdir -p /app/logs && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose the enhanced admin server port
EXPOSE 8001

# Run gunicorn with uvicorn worker
CMD ["gunicorn", "tubesensei.app.main_enhanced:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8001", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
