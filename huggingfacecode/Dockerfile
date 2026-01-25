# Dockerfile for NL-to-NoSQL Converter on HuggingFace Spaces
# Fixed for proper Streamlit routing

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app_server.py .
COPY schema_examples.json .
COPY src/ ./src/
COPY qwen_text2sql_adapter/ ./qwen_text2sql_adapter/
COPY sql_to_mongodb_adapter/ ./sql_to_mongodb_adapter/

# Create cache directories
RUN mkdir -p /tmp/transformers_cache /tmp/hf_home

# Environment variables
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/hf_home
ENV PYTHONUNBUFFERED=1

# Expose port 7860 (HuggingFace standard)
EXPOSE 7860

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# Create startup script - Run everything on port 7860
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Kill any existing processes\n\
fuser -k 7860/tcp 2>/dev/null || true\n\
fuser -k 8000/tcp 2>/dev/null || true\n\
sleep 2\n\
\n\
echo "Starting backend server on port 8000..."\n\
python app_server.py --port 8000 &\n\
BACKEND_PID=$!\n\
echo "Backend PID: $BACKEND_PID"\n\
\n\
echo "Waiting for backend to be ready..."\n\
for i in {1..45}; do\n\
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then\n\
        echo "Backend is ready!"\n\
        break\n\
    fi\n\
    if [ $i -eq 45 ]; then\n\
        echo "Backend failed to start in 90 seconds"\n\
        kill $BACKEND_PID 2>/dev/null || true\n\
        exit 1\n\
    fi\n\
    echo "Waiting for backend... ($i/45)"\n\
    sleep 2\n\
done\n\
\n\
# Wait a bit more to ensure backend is fully ready\n\
sleep 3\n\
\n\
echo "Starting Streamlit on port 7860 (main port)..."\n\
streamlit run src/streamlit_app.py \\\n\
    --server.port=7860 \\\n\
    --server.address=0.0.0.0 \\\n\
    --server.headless=true \\\n\
    --server.enableCORS=false \\\n\
    --server.enableXsrfProtection=false &\n\
STREAMLIT_PID=$!\n\
echo "Streamlit PID: $STREAMLIT_PID"\n\
\n\
# Wait for both processes\n\
wait -n\n\
\n\
# If either exits, kill the other\n\
kill $BACKEND_PID $STREAMLIT_PID 2>/dev/null || true\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
