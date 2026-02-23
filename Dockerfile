FROM python:3.11-slim

WORKDIR /app

# System dependencies for document parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir ".[postgres]"

COPY . .

EXPOSE 8000

CMD ["python", "run_server.py"]
