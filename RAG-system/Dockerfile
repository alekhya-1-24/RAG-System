FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/       ./src/
COPY static/    ./static/
COPY build_index.py .

# Note: data/ directory (rag_index.pkl, persona.json, messages.json)
# must be built locally and copied here, OR built at container startup.
# For HF Spaces: build locally, commit data/ to the repo.
COPY data/      ./data/

EXPOSE 7860

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]
