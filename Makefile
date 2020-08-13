SHELL := /bin/bash

setup:
	sudo npm i -g editorconfig
	python3 -m venv venv
	source ./venv/bin/activate
	pip install -r requirements.txt

build:
	source ./venv/bin/activate
	make clean
	make error_regions

clean:
	source ./venv/bin/activate
	if [ -d "./data/cleandata/Info pluviometricas" ]; then rm -r "./data/cleandata/Info pluviometricas"; fi
	python ./src/Pipeline/clean_infopluviometricas.py

error_regions:
	source ./venv/bin/activate
	python ./src/Pipeline/error_regions.py

commit:
	python pythonify.py

upload_bq:
	python ./src/Pipeline/upload_bigquery.py

download_bq:
	python ./src/Pipeline/download_bigquery.py
