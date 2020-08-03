SHELL := /bin/bash

setup:
	python3 -m venv venv
	source ./venv/bin/activate
	pip install -r requirements.txt

build:
	source ./venv/bin/activate
	make clean
	make get_regions

clean:
	source ./venv/bin/activate
	if [ -d "./data/cleandata/Info pluviometricas" ]; then rm -r "./data/cleandata/Info pluviometricas"; fi
	python ./develop/Pipeline/clean_infopluviometricas.py

get_regions:
	source ./venv/bin/activate
	python ./develop/Pipeline/get_regions.py
