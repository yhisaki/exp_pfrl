SHELL=/bin/bash

.PHONY: setup venv clean docker format singularity

venv:
	python3 -m venv .venv
	source .venv/bin/activate
	pip install -r requirements.txt

docker:
	docker build -t exp_pfrl .

singularity: docker
	singularity build exp_pfrl.sif Singularity

format:
	isort .
	black .