PY ?= .venv/bin/python

.PHONY: setup experiments evaluate dashboard test all clean

setup:            ## create .venv and install dependencies
	uv venv .venv --python 3.13
	uv pip install --python $(PY) -r requirements.txt

experiments:      ## train all models under both protocols, cache traces (~6 min on a laptop GPU)
	$(PY) scripts/run_experiments.py --dataset ALL

evaluate:         ## tables, sweeps, ablations -> artifacts/results/results.json
	$(PY) scripts/evaluate.py

dashboard:        ## export traces for the offline console, verify its JS replay against Python
	$(PY) scripts/export_dashboard.py
	node scripts/check_dashboard.mjs

test:
	$(PY) -m pytest -q tests

all: experiments evaluate dashboard test

clean:
	rm -rf artifacts/traces artifacts/models artifacts/results
