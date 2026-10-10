# Serving image: API and dashboard over the committed results. CPU only, no training dependencies.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 GRIDCAST_ROOT=/app
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve,copilot]"

COPY app ./app
COPY data/processed ./data/processed
COPY results/tables ./results/tables
COPY results/serving ./results/serving

RUN useradd --create-home gridcast && chown -R gridcast /app
USER gridcast
EXPOSE 8000 8501
CMD ["uvicorn", "gridcast.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
