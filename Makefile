SHELL := /bin/bash

setup:
	git config --local core.hooksPath .githooks/
	npm i -g editorconfig
	pip install -r requirements.txt

build:
	make clean
	make error_regions
	make repair_data
	# make prep_data

clean:
	python ./src/Pipeline/clean_infopluviometricas.py
	python ./src/Pipeline/clean_owm_history_bulk.py

error_regions:
	python ./src/Pipeline/error_regions.py

repair_data:
	python ./src/Pipeline/repair_regions.py
	python ./src/Pipeline/get_labels_day.py
	python ./src/Pipeline/get_labels_hour.py

prep_data:
	python ./src/Pipeline/prep_data.py

upload_bq:
	python ./src/Pipeline/upload_bigquery.py

download_bq:
	python ./src/Pipeline/download_bigquery.py
