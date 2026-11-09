#!/bin/sh
set -eu

mkdir -p /app/data/pdfs

if [ ! -f /app/data/rag_database.db ] && [ -f /opt/seed/rag_database.db ]; then
    cp /opt/seed/rag_database.db /app/data/rag_database.db
fi

if ! find /app/data/pdfs -maxdepth 1 -type f -name '*.pdf' -print -quit | grep -q .; then
    cp /opt/seed/pdfs/*.pdf /app/data/pdfs/
fi

until curl -sf http://qdrant:6333/healthz >/dev/null; do
    sleep 1
done
python -c "from src.ingestion import rebuild_vector_index_from_db; print(rebuild_vector_index_from_db())"

python -m uvicorn src.app:app --host 0.0.0.0 --port 8000 &
exec streamlit run src/ui.py --server.port 8501 --server.address 0.0.0.0
