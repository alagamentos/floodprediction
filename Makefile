SHELL := /bin/bash

setup:
	git config --local core.hooksPath .githooks/
	sudo npm i -g editorconfig
	python3 -m venv venv
	source ./venv/bin/activate
	pip install -r requirements.txt

build:
	source ./venv/bin/activate
	make clean
	make error_regions
	make repair_regions

clean:
	source ./venv/bin/activate
	python ./src/Pipeline/clean_infopluviometricas.py

error_regions:
	source ./venv/bin/activate
	python ./src/Pipeline/error_regions.py

repair_regions:
	source ./venv/bin/activate
	python ./src/Pipeline/repair_regions.py

commit:
	source ./venv/bin/activate
	python pythonify.py

upload_bq:
	source ./venv/bin/activate
	python ./src/Pipeline/upload_bigquery.py

download_bq:
	source ./venv/bin/activate
	python ./src/Pipeline/download_bigquery.py
