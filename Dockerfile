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

# Copy application files
COPY . .

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

# Entrypoint script starts both backend and frontend
CMD ["sh", "-c", "python -m uvicorn src.app:app --host 0.0.0.0 --port 8000 & exec streamlit run src/ui.py --server.port 8501 --server.address 0.0.0.0"]
