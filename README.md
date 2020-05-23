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

---
## Make
```bash
make clean
```
Deleta todos os dados existentes de data/cleandata/Info pluviometricas e executa  `clean_infopluviometricas.py`- duas pastas em cleandata são criadas:
 - **Merged Data**: Contém um arquivo único com todos os dados de info pluvimétrica.
 - **Concatanated**:  Para cada estação disponível cria um arquivo com todos os dados concatenados.

 ---
```bash
make get_regions
```
Acha as regiões com erro dos dados *info pluviométrica*. Cria um novo arquivo em data/cleandata/Merged Data/**merged_wRegions.csv** 

---
```bash
make all
```
Instala os pacotes necessários (pip - comando abaixo) e executa todos os makes anterirores.

```bash
pip install -r requirements.txt
```


