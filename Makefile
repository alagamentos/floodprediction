SHELL := /bin/bash

setup:
	npm i -g editorconfig
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
	python ./src/Pipeline/clean_infopluviometricas.py

get_regions:
	source ./venv/bin/activate
	python ./src/Pipeline/get_regions.py

commit:
	python pythonify.py
