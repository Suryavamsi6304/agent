FROM python:3.11-slim

WORKDIR /app

# ── Corporate SSL certificate support (Zscaler, etc.) ─────────────────────────
# To inject your corporate CA certificate at build time:
#   docker build --build-arg CORPORATE_CA_CERT="$(cat /path/to/zscaler-ca.crt)" .
# Or mount it at runtime via an env var:
#   docker run -e REQUESTS_CA_BUNDLE=/certs/ca.crt -v /path/to/ca.crt:/certs/ca.crt ...
ARG CORPORATE_CA_CERT=""
RUN if [ -n "$CORPORATE_CA_CERT" ]; then \
        echo "$CORPORATE_CA_CERT" > /usr/local/share/ca-certificates/corporate-ca.crt && \
        update-ca-certificates; \
    fi

# Install system dependencies for PyMuPDF and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/lists/*

# Install Python dependencies
# If behind a corporate proxy, pass --trusted-host flags or set pip env vars
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
