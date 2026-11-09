FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Keep seed data outside the runtime volume so a fresh named volume can be
# initialized without losing the starter corpus.
COPY data/rag_database.db /opt/seed/rag_database.db
COPY starter/data/pdfs /opt/seed/pdfs
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

# Copy application files. Secrets are excluded by .dockerignore.
COPY . .

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

# Entrypoint script starts both backend and frontend
ENTRYPOINT ["docker-entrypoint.sh"]
