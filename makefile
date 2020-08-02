setup:
	python3 -m venv venv
	source ./venv/bin/activate
	pip install -r requirements.txt

all:
	make clean
	make get_regions

clean:
	if [ -d "./data/cleandata/Info pluviometricas" ]; then rm -r "./data/cleandata/Info pluviometricas"; fi
	python ./develop/Pipeline/clean_infopluviometricas.py

get_regions:
	python ./develop/Pipeline/get_regions.py
	