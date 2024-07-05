# TrialDura
Code for the Paper: [TrialDura: Hierarchical Attention Transformer for Interpretable Clinical Trial Duration Prediction](https://arxiv.org/pdf/2404.13235)

## Installation
```
conda create -n clinical python==3.8
conda activate clinical
pip install rdkit
pip install tqdm scikit-learn seaborn numpy pandas icd10-cm
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install conda-forge::transformers
pip install tensorboard
conda install lightgbm
conda install xgboost
```

see also:

Operating System: Debian 6.1.0-18-amd64
Kernel Version: 6.1.76-1 (2024-02-01)
Architecture: x86_64
conda 24.3.0
Python 3.8.19
pytorch 1.12.1
torchvision 0.13.1
torchaudio 0.12.1



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
run preprocess/generate_target.ipynb

### DrugBank
1. Apply the Drugbank license https://go.drugbank.com/releases/latest
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
Note that this process is heavy and may take some time to complete. It's recommended to run it in the background.
```
python preprocess/protocol_encode.py
```


## Model
Run model/run_NN.ipynb

You can visualization the results using tensorboard, the logs are located under logs/
