# Instituto Mauá de Tecnologia - Prefeitura de Santo André

## TCC - Sistema Inteligente de Previsão de Alagamentos

### Integrantes

Nome | RA | GitHub
------------ | ------------- | -------------
Felipe Ippolito | 12.01378-0 | [feippolito](https://github.com/feippolito)
Felipe Andrade | 15.00175-0 | [Kaisen-san](https://github.com/Kaisen-san)
Vinícius Pereira | 16.03343-4 | [VinPer](https://github.com/VinPer)

### Orientador

Prof. Me. Tiago Sanches ([Tiagoeem](https://github.com/Tiagoeem))

### Estrutura do projeto

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

### Requerimentos

É necessário ter instalado os seguintes programas:

- [Python3](https://www.python.org/downloads)
- [Jupyter Notebook](https://jupyter.org/install)
- [NodeJS](https://nodejs.org/en/download)
- [Make](http://gnuwin32.sourceforge.net/packages/make.htm) *(já vem instalado no Linux)*

Caso queira utilizar o [Virtual Environment](https://docs.python.org/3/tutorial/venv.html) do *Python*, rode os seguintes comandos na pasta raíz do projeto:

```bash
python -m venv venv # Cria o virtual environment
source ./venv/bin/activate # Ativa o virtual environment
```

> Após a instalação dos programas, rode o comando `make setup` na pasta raíz do projeto

#### BigQuery

Para acessar os dados no [BigQuery](https://cloud.google.com/bigquery) é necessário ter a chave de acesso do projeto criado na plataforma. Isso não impacta o uso do projeto localmente, porém será necessário executar a limpeza dos dados através do `make build`, o que demora cerca 1h30 (a depender do poder de processamento do computador), além de não ser possível rodar os comandos `make upload_bq` e `make download_bq`. Caso tenha interesse, solicite a chave de acesso aos integrantes do grupo.

> Uma vez com o arquivo *key.zip* em mãos, extraia-o e copie e cole a pasta *key* para dentro da pasta raíz do projeto.

### Comandos Make

Configura o ambiente local e instala todas as dependências necessárias

```bash
make setup
```

---

Executa os arquivos *clean_infopluviometricas.py* e o *clean_owm_history_bulk.py* a fim de limpar e padronizar os dados a serem utilizados. Dessa execução são geradas três pastas em *cleandata*:
 - **Info pluviometricas/Merged Data**: Contém um único arquivo com todos os dados de todas as estações concatenados, limpos e formatados.
 - **Info pluviometricas/Concatenated**: Contém um arquivo para cada estação com todos os dados da mesma concatenados, limpos e formatados.
 - **OpenWeather**: Contém um único arquivo com todos os dados do OpenWeatherMap limpos e formatados.

```bash
make clean
```

 ---

Acha as regiões com erro nos dados das informações pluviométricas e cria um novo arquivo em *data/cleandata/Merged Data/**error_regions.csv***

```bash
make error_regions
```

---
Executa os arquivos *repair_regions.py*, *get_labels_day.py* e *get_labels_hour.py* -- Correção dos dados de info pluviometricas e ordens de serviço.
- **repair_regions.py**: Interpola os dados das regiões de erro e depois aplica o regressor XGBoost nos mesmos. Um novo arquivo é criado em *data/cleandata/Merged Data/**repaired.csv***.
-  **get_labels_day.py** e **get_labels_hour.py**: Corrige as labels de ordens de serviço de acordo com o máximo local do número de ordens de serviço e filtrar de acordo com a precipitação.  Dois arquivos são criados em *data/cleandata/Ordens de serviço/**labels_day.csv*** e *data/cleandata/Ordens de serviço/**labels_hour.csv***.

```bash
make repair_data
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

### Referências
- [Artigos utilizados na revisão bibliográfica](https://drive.google.com/drive/folders/1RDT4sAvsjU82O3m3slLdigGo8T5wgxBc?usp=sharing)
