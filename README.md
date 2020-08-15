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

Para ter acesso aos dados no BigQuery é necessário ter a chave de acesso do projeto criado na plataforma. Isso não impacta o uso do projeto localmente, porém será necessário executar a limpeza dos dados através do `make build`, o que demora cerca 1h30 (a depender do poder de processamento do computador). Solicite a chave de acesso aos integrantes do grupo, e uma vez com o arquivo .zip em mãos, extraia-o e copie e cole a pasta *key* para dentro da pasta raiz do projeto.

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

Acha as regiões com erro nos dados das informações pluviométricas e cria um novo arquivo em *data/cleandata/Merged Data/**error_regions.csv***

```bash
make error_regions
```

---

Interpola os dados das regiões de erro e depois aplica o regressor XGBoost nos mesmos. Um novo arquivo é criado em *data/cleandata/Merged Data/**repaired.csv***

```bash
make repair_regions
```

---

Executa todos os makes relacionados a manipulação de dados (*clean*, *error_regions* e *repair_regions*)

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
