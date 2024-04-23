# TrialDura
Code for the Paper: TrialDura: Hierarchical Attention Transformer for Interpretable Clinical Trial Duration Prediction

## Installation
```
conda create -n clinical python==3.8
conda activate clinical
pip install rdkit
pip install tqdm scikit-learn seaborn numpy pandas icd10-cm
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install conda-forge::transformers
pip install tensorboard
```


## Preprocess

### ClinicalTrial.gov

```
cd data/raw
wget https://clinicaltrials.gov/AllPublicXML.zip
unzip AllPublicXML.zip -d ../trials
cd ..

find trials/* -name "NCT*.xml" | sort > trials/all_xml.txt
```

### Generate Target Time
open preprocess/generate_target.ipynb and run it

### DrugBank
1. Apply the Drugbank license
2. Download the data(full database.xml) in data/raw directory.
3. run preprocess/drugbank_xml2csv.ipynb

### Disease
```
python preprocess/collect_disease_from_raw.py
```


### Selection criteria of clinical trial
``` 
python preprocess/collect_raw_data.py | tee data_process.log 

python preprocess/collect_input_data.py
```

### Data Split
```
python preprocess/data_split.py 
```

### Sentence to embedding
```
python preprocess/protocol_encode.py
```

## Model
Run model/run_NN.ipynb

You can visualization the results using tensorboard, the logs are located under logs/
