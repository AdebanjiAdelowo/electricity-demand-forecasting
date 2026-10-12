PY ?= uv run python
.PHONY: setup data test lint baselines deep report studies serving figures api dashboard all

setup:            ## create the project environment from the lockfile
	uv sync --extra deep --extra serve --extra copilot --extra dev

data:             ## download EIA-930 and Open-Meteo data, build the processed dataset
	$(PY) -m gridcast.cli data

test:
	GRIDCAST_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $(PY) -m pytest --ignore=tests/test_deep.py
	GRIDCAST_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $(PY) -m pytest tests/test_deep.py

lint:
	uv run ruff check src app tests experiments

baselines:        ## statistical and machine-learning models, CPU only
	$(PY) -m gridcast.cli benchmark --group classical
	$(PY) -m gridcast.cli benchmark --group external

deep:             ## neural models (MPS, CUDA or CPU)
	$(PY) -m gridcast.cli benchmark --group deep

report:
	$(PY) -m gridcast.cli report
	$(PY) -m gridcast.cli probabilistic

studies:
	$(PY) -m gridcast.cli study ablation
	$(PY) -m gridcast.cli study robustness
	$(PY) -m gridcast.cli study stress
	$(PY) -m gridcast.cli study lookback
	$(PY) -m gridcast.cli study size
	$(PY) -m gridcast.cli anomaly

serving:
	$(PY) -m gridcast.cli serving
	$(PY) -m gridcast.cli copilot-eval

figures:
	$(PY) -m gridcast.cli figures

api:
	uv run uvicorn gridcast.api.main:app --port 8000

dashboard:
	uv run streamlit run app/dashboard.py

all: data baselines deep report studies serving figures
