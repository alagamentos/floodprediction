# TCC - Sistema Inteligente de Previsão de Alagamentos

## Integrantes
Nome | RA | GitHub
------------ | ------------- | -------------
Felipe Ippolito | 12.01378-0 | [feippolito](https://github.com/feippolito)
Felipe Andrade | 15.00175-0 | [Kaisen-san](https://github.com/Kaisen-san)
Vinícius Pereira | 16.03343-4 | [VinPer](https://github.com/VinPer)

## Estrutura do projeto

```
.
├── data
│   ├── cleandata
│   ├── rawdata
├── src
│   ├── Notebooks
│   ├── Pipeline
│   ├── PowerBI
├── Makefile
├── README.md
├── requirements.txt
```

## Requerimentos

É necessário ter instalado na máquina os seguintes programas:

- Python3
- Jupyter notebook
- NodeJS

## Comandos Make

Instala o Virtual Environment do Python, junto com todos os pacotes necessários

```bash
make setup
```

---

Gera os arquivos Python a partir dos Jupyter notebooks existentes afim de facilitar o rastreio das mudanças. **DEVE SER EXECUTADO ANTES DE CADA COMMIT!**

```bash
make commit
```

---

Deleta todas as pastas existentes em *data/cleandata/Info pluviometricas* e executa o arquivo `clean_infopluviometricas.py`. Dessa execução são geradas duas pastas em *cleandata*:
 - **Merged Data**: Contém um arquivo único com todos os dados das informações pluvimétricas.
 - **Concatenated**: Contém um arquivo para cada estação com todos os dados da mesma concatenados.

```bash
make clean
```

 ---

Acha as regiões com erro nos dados das informações pluviométricas e cria um novo arquivo em *data/cleandata/Merged Data/**merged_wRegions.csv***

```bash
make error_regions
```

---

Executa todos os makes relacionados a manipulação de dados (*clean* e *error_regions*)

```bash
make build
```

---

Realiza o upload dos dados dos arquivos de *merged.csv*, *error_regions.csv* e *repaired.csv* da pasta *data/cleandata/Info pluviometricas/Merged Data* para o BigQuery

```bash
make upload_bq
```

---

Realiza o download dos dados no BigQuery para os arquivos de *merged.csv*, *error_regions.csv* e *repaired.csv* na pasta *data/cleandata/Info pluviometricas/Merged Data*.

```bash
make download_bq
```

## Referências
- [Artigos utilizados na revisão bibliográfica](https://drive.google.com/drive/folders/1RDT4sAvsjU82O3m3slLdigGo8T5wgxBc?usp=sharing)
