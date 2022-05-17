SHELL := /bin/bash

setup:
	git config --local core.hooksPath .githooks/
	npm i -g editorconfig
	python3 -m pip install -r requirements.txt

build:
	make clean
	make error_regions
	make repair_data
	# make prep_data

clean:
	python3 ./src/Pipeline/clean_infopluviometricas.py
	python3 ./src/Pipeline/clean_owm_history_bulk.py

error_regions:
	python3 ./src/Pipeline/error_regions.py

repair_data:
	# python3 ./src/Pipeline/repair_regions.py
	python3 ./src/Pipeline/get_labels_day.py
	python3 ./src/Pipeline/get_labels_hour.py

prep_data:
	python3 ./src/Pipeline/prep_data.py

upload_bq:
	# python3 ./src/Pipeline/upload_bigquery.py

download_bq:
	python3 ./src/Pipeline/download_bigquery.py
