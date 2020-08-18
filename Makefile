SHELL := /bin/bash

setup:
	git config --local core.hooksPath .githooks/
	npm i -g editorconfig
	pip install -r requirements.txt

build:
	make clean
	make error_regions
	make repair_regions

clean:
	python ./src/Pipeline/clean_infopluviometricas.py
	python ./src/Pipeline/clean_owm_history_bulk.py

error_regions:
	python ./src/Pipeline/error_regions.py

repair_regions:
	python ./src/Pipeline/repair_regions.py

upload_bq:
	python ./src/Pipeline/upload_bigquery.py

download_bq:
	python ./src/Pipeline/download_bigquery.py
