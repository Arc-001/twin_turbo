PY ?= .venv/bin/python

.PHONY: setup experiments evaluate test all clean

setup:            ## create .venv and install dependencies
	uv venv .venv --python 3.13
	uv pip install --python $(PY) -r requirements.txt

experiments:      ## train all models under both protocols, cache traces (~6 min on a laptop GPU)
	$(PY) scripts/run_experiments.py --dataset ALL

evaluate:         ## tables, sweeps, ablations -> artifacts/results/results.json
	$(PY) scripts/evaluate.py

test:
	$(PY) -m pytest -q tests

all: experiments evaluate test

clean:
	rm -rf artifacts/traces artifacts/models artifacts/results
