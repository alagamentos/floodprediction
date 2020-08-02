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
├── develop
│   ├── Notebooks 
│   ├── Pipeline
│   ├── PowerBI
├── makefile
├── README.md
├── requirements.txt
```

## Make commands

Instala o Virtual Environment do Python, junto com todos os pacotes necessários

```bash
make setup
```

---

Deleta todos os dados existentes de *data/cleandata/Info pluviometricas* e executa `clean_infopluviometricas.py`. Duas pastas em *cleandata* são criadas:
 - **Merged Data**: Contém um arquivo único com todos os dados de info pluvimétrica.
 - **Concatanated**:  Para cada estação disponível cria um arquivo com todos os dados concatenados.

```bash
make clean
```

 ---

Acha as regiões com erro dos dados *info pluviométrica*. Cria um novo arquivo em *data/cleandata/Merged Data/**merged_wRegions.csv***

```bash
make get_regions
```

---

Executa todos os makes anteriores

```bash
make all
```
