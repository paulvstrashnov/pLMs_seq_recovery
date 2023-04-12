# pLMs_seq_recovery


### Данные (~250 Mb, ~569k последовательностей): 
```bash
wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
```

### Сформировать свой датасет:
```bash
python prepare_dataset.py
```

### Установить в Conda environment
```bash
conda install mamba -n base -c conda-forge
mamba env create -f env.yml
conda activate pLM
```

### Usage
```bash
python main.py --models ESM2_8M --data_path data/sample-100000_min128_max384.fasta
```
Для того, чтобы запустить оценку пар энкодер/декодер для нескольких моделей, их имена нужно перечислить через запятую.
```bash
python main.py --models ESM2_8M,ProtBERTModel --data_path data/sample_28356.fasta --batch_size 32 --num_workers 4 --log_file ESM2_8M-ProtBERTModel.log --device cuda:1
```
|Model|Embedding dim|
|---|:-:|
|ESM2_8M|320|
|ESM2_35M|480|
|ESM2_150M|640|
|ESM2_650M|1280|
|ESM2_3B|2560|
|ESM2_15B|5120|
|ProtBERT|1024|

## Results
- Mean Levenstein - среднее расстояние Левенштейна на датасете.
- Mean Normalized Levenstein - среднее расстояние Левенштейна, нормализованное на длину сиквенса, на датасете.
- dNW - среднее разности (NW - NW-self) на датасете. NW - скор алгоритма Нидлмана-Вунша (BLOSUM62) между входной и декодированной последовательностями. NW_self - скор алгоритма Нидлмана-Вунша (BLOSUM62) для выравнивания входной последовательностью c самой собой.

В идеальном случае, все метрики должны дать 0. Чем ниже абсолютные значения метрик, тем лучше.


### Результаты для датасета sample_28356.fasta
sample_28356.fasta - 5% от Uniprot/SwissProt, длина сиквенсов ограничена 256, отсутсвуют нестандартные аминокислоты.
|Model|Mean Levenstein|Mean Normalized Levenstein|dNW|
|---|:-:|:-:|:-:|
|Prot_T5|0.0|0.0|0.0|
|ESM2_8M|3.27|0.01|-26.00|
|ESM2_8M_dec*|3.33|0.02|-29.65|
|ESM2_150M_dec*|3.66|0.02|-33.35|
|ESM2_35M_dec*|3.90|0.02|-36.24|
|ESM2_650M_dec*|4.99|0.03|-45.51|
|ESM2_35M|5.72|0.03|-45.28|
|ESM2_150M|6.31|0.03|-45.35|
|ESM2_650M|7.89|0.04|-54.26|
|ProtBERT|7.97|0.04|-57.55|
|ESM2_3B**|9.24|0.04|-62.13|
|ESM2_3B_dec**|14.57|0.08|-126.63|
|Electra|92.46|0.44|-467.33|


*_dec - модели ESM в режиме is_decoder=True
** - ESM2_3B иногда предсказывает U аминокислоту, которая сейчас просто удаляется из последовательности уменьшая ее. Это похоже сильно влияет на скор выравнивания (dNW) из-за появления гепа 
