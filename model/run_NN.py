# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

from models import Protocol_Embedding_Regression, Protocol_Attention_Regression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from transformers import AutoTokenizer, AutoModel
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


# %%
import sys
sys.path.append('../')

from preprocess.protocol_encode import protocol2feature, load_sentence_2_vec, get_sentence_embedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
import os
os.getcwd()

use_valid = True

# %%
sentence2vec = load_sentence_2_vec("../data") 

# %%
train_data = pd.read_csv(f'../data/time_prediction_train.csv', sep='\t')
test_data = pd.read_csv(f'../data/time_prediction_test.csv', sep='\t')

if use_valid:
    train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=0)
print(train_data.head())

# %%
# Missing Value Handling
train_data['criteria'].fillna('', inplace=True)
test_data['criteria'].fillna('', inplace=True)

# %%
# # 32 sentences length can cover 95% of the data 

# criteria_lst = train_data['criteria']

# in_criteria_lengths = []
# ex_criteria_lengths = []

# for criteria in criteria_lst:
#     in_criteria, ex_criteria = protocol2feature(criteria, sentence2vec)
#     in_criteria_lengths.append(len(in_criteria))
#     ex_criteria_lengths.append(len(ex_criteria))

# print(f"Inclusion: {pd.Series(in_criteria_lengths).describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99, 0.999])}")
# print(f"Exclusion: {pd.Series(ex_criteria_lengths).describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99, 0.999])}")


# %%

def pad_sentences(paragraphs, padding_size):
    padded_paragraphs = []
    mask_matrices = []

    for p in paragraphs:
        num_padding = padding_size - p.size(0)
        
        if num_padding > 0:
            padding = torch.zeros(num_padding, p.size(1))
            padded_p = torch.cat([p, padding], dim=0)
        else:
            padded_p = p

        # 1 for actual data, 0 for padding
        mask = torch.cat([torch.ones(p.size(0)), torch.zeros(num_padding)], dim=0)

        padded_paragraphs.append(padded_p)
        mask_matrices.append(mask)

    padded_paragraphs_tensor = torch.stack(padded_paragraphs)
    mask_matrices_tensor = torch.stack(mask_matrices)

    return padded_paragraphs_tensor, mask_matrices_tensor


def criteria2embedding(criteria_lst, padding_size):
    criteria_lst = [protocol2feature(criteria, sentence2vec) for criteria in criteria_lst]

    max_sentences = max(max(p[0].size(0), p[1].size(0)) for p in criteria_lst)
    if max_sentences < padding_size:
        print(f"Warning: padding size is larger than the maximum number of sentences in the data. Padding size: {padding_size}, Max sentences: {max_sentences}")

    incl_criteria = [criteria[0][:padding_size] for criteria in criteria_lst]
    incl_emb, incl_mask = pad_sentences(incl_criteria, padding_size)

    excl_criteria = [criteria[1][:padding_size] for criteria in criteria_lst]
    excl_emb, excl_mask = pad_sentences(excl_criteria, padding_size)

    return incl_emb, incl_mask, excl_emb, excl_mask


padding_size = 32
incl_emb = {}
incl_mask = {}
excl_emb = {}
excl_mask = {}

incl_emb['train'], incl_mask['train'], excl_emb['train'], excl_mask['train'] = criteria2embedding(train_data['criteria'], padding_size)
incl_emb['test'], incl_mask['test'], excl_emb['test'], excl_mask['test'] = criteria2embedding(test_data['criteria'], padding_size)


# %%
encoder = OneHotEncoder(sparse=False)
encoder.fit(train_data[['phase']])

phase_emb = {}
phase_emb['train'] = torch.tensor(encoder.transform(train_data[['phase']])).float()
phase_emb['test'] = torch.tensor(encoder.transform(test_data[['phase']])).float()



# %%
def drug2embedding(drug_lst):
    model_name = "dmis-lab/biobert-base-cased-v1.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
     
    drug_emb = []
    for drugs in tqdm(drug_lst):
        if len(drugs) == 0:
            print("Warning: Empty drug list is found")
            drug_emb.append(torch.zeros(768, dtype=torch.float32))
        else:
            # mean pooling
            drugs_emb = torch.mean(torch.stack([get_sentence_embedding(drug, tokenizer, model) for drug in drugs.split(';')]), dim=0)
            drug_emb.append(drugs_emb)
    
    return torch.stack(drug_emb)

def disease2embedding(disease_lst):
    model_name = "dmis-lab/biobert-base-cased-v1.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
     
    disease_emb = []
    for diseases in tqdm(disease_lst):
        if len(diseases) == 0:
            print("Warning: Empty disease list is found")
            disease_emb.append(torch.zeros(768, dtype=torch.float32))
        else:
            # mean pooling
            diseases_emb = torch.mean(torch.stack([get_sentence_embedding(disease, tokenizer, model) for disease in diseases.split(';')]), dim=0)
            disease_emb.append(diseases_emb)
    
    return torch.stack(disease_emb)


# %%
if not os.path.exists('../data/drug_emb.pt') or not os.path.exists('../data/disease_emb.pt'):
    drug_emb = {}
    drug_emb['train'] = drug2embedding(train_data['drugs'].tolist())
    drug_emb['test'] = drug2embedding(test_data['drugs'].tolist())

    disease_emb = {}
    disease_emb['train'] = disease2embedding(train_data['diseases'].tolist())
    disease_emb['test'] = disease2embedding(test_data['diseases'].tolist())

    if use_valid:
        drug_emb['valid'] = drug2embedding(valid_data['drugs'].tolist())
        disease_emb['valid'] = disease2embedding(valid_data['diseases'].tolist())

    torch.save(drug_emb, '../data/drug_emb.pt')
    torch.save(disease_emb, '../data/disease_emb.pt')
else:
    drug_emb = torch.load('../data/drug_emb.pt')
    disease_emb = torch.load('../data/disease_emb.pt')

# %%
class Trial_Dataset(Dataset):
	def __init__(self, nctid_lst, incl_emb, incl_mask, excl_emb, excl_mask, drug_emb, dis_emb, phase_emb, target_lst):
		self.nctid_lst = nctid_lst 
		self.target_lst = torch.tensor(target_lst.values, dtype=torch.float32)

		self.incl_emb = incl_emb
		self.incl_mask = incl_mask

		self.excl_emb = excl_emb
		self.excl_mask = excl_mask
		self.drug_emb = drug_emb
		self.dis_emb = dis_emb
		self.phase_emb = phase_emb

	def __len__(self):
		return len(self.nctid_lst)

	def __getitem__(self, idx):
		return self.nctid_lst.iloc[idx], (self.incl_emb[idx], self.incl_mask[idx]), (self.excl_emb[idx], self.excl_mask[idx]), self.drug_emb[idx], self.dis_emb[idx], self.phase_emb[idx], self.target_lst[idx]


# %%
batch_size = 256

train_dataset = Trial_Dataset(train_data['nctid'], incl_emb['train'], incl_mask['train'], excl_emb['train'], excl_mask['train'], drug_emb['train'], disease_emb['train'], phase_emb['train'], train_data['time_day'])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = Trial_Dataset(test_data['nctid'], incl_emb['test'], incl_mask['test'], excl_emb['test'], excl_mask['test'], drug_emb['test'], disease_emb['test'], phase_emb['test'], test_data['time_day'])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# %%
if use_valid:
    valid_data['criteria'].fillna('', inplace=True)
    incl_emb['valid'], incl_mask['valid'], excl_emb['valid'], excl_mask['valid'] = criteria2embedding(valid_data['criteria'], padding_size)
    phase_emb['valid'] = torch.tensor(encoder.transform(valid_data[['phase']])).float()
    valid_dataset = Trial_Dataset(valid_data['nctid'], incl_emb['valid'], incl_mask['valid'], excl_emb['valid'], excl_mask['valid'], drug_emb['valid'], disease_emb['valid'], phase_emb['valid'], valid_data['time_day'])
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


# %%
def test(model, data_loader):
    model.eval()

    with torch.no_grad():
        predictions = []
        targets = []
        
    for batch_idx, batch_data in enumerate(data_loader):
            nctid, (inclusion_emb, inclusion_mask), (exclusion_emb, exclusion_mask), drug_emb, disease_emb, phase_emb, target = batch_data
            inclusion_emb, inclusion_mask, exclusion_emb, exclusion_mask, drug_emb, disease_emb, phase_emb, target = inclusion_emb.to(device), inclusion_mask.to(device), exclusion_emb.to(device), exclusion_mask.to(device), drug_emb.to(device), disease_emb.to(device), phase_emb.to(device), target.to(device)

            output = model.forward(inclusion_emb, inclusion_mask, exclusion_emb, exclusion_mask, drug_emb, disease_emb, phase_emb)
            prediction = output[:, 0]

            predictions.extend(prediction.tolist())
            targets.extend(target.tolist())

    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    pearson_score, _ = pearsonr(targets, predictions)

    return mae, mse, r2, pearson_score


# %%
mae_list = []
mse_list = []
r2_list = []
pearson_list = []
for i in range(5):
    num_epochs = 15

    # protocol_model = Protocol_Embedding_Regression(output_dim=1)
    torch.manual_seed(i)
    protocol_model = Protocol_Attention_Regression(sentence_embedding_dim=768, linear_output_dim=1, transformer_encoder_layers=2, num_heads=8,  dropout=0.1, pooling_method="cls")
    print(protocol_model)

    protocol_model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(protocol_model.parameters(), lr=0.029, weight_decay=0.00276)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)

    # Create a SummaryWriter instance
    with SummaryWriter(f'logs/NN_model_log') as writer:
        print("Start training")
        best_mse = float('inf')
        for epoch in tqdm(range(num_epochs)):
            protocol_model.train()
            for batch_idx, batch_data in enumerate(train_loader):
                nctid, (inclusion_emb, inclusion_mask), (exclusion_emb, exclusion_mask), drug_emb, disease_emb, phase_emb, target = batch_data
                inclusion_emb, inclusion_mask, exclusion_emb, exclusion_mask, drug_emb, disease_emb, phase_emb, target = inclusion_emb.to(device), inclusion_mask.to(device), exclusion_emb.to(device), exclusion_mask.to(device), drug_emb.to(device), disease_emb.to(device), phase_emb.to(device), target.to(device)

                output = protocol_model.forward(inclusion_emb, inclusion_mask, exclusion_emb, exclusion_mask, drug_emb, disease_emb, phase_emb)
                prediction = output[:, 0]
                loss = criterion(prediction, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Write the loss to TensorBoard
                if batch_idx % 50 == 0:
                    writer.add_scalar('Loss', loss.item(), epoch * len(train_loader) + batch_idx)
            
            if epoch % 1 == 0:
                if use_valid:
                    valid_mae, valid_mse, _, _ = test(protocol_model, valid_loader)
                    writer.add_scalar('valid_MAE', valid_mae, epoch)
                    writer.add_scalar('valid_MSE', valid_mse, epoch)
                else:
                    test_mae, test_mse, _, _ = test(protocol_model, test_loader)
                    writer.add_scalar('MAE', test_mae, epoch)
                    writer.add_scalar('MSE', test_mse, epoch)

                train_mae, train_mse, _, _ = test(protocol_model, train_loader)
                writer.add_scalar('train_MAE', train_mae, epoch)
                writer.add_scalar('train_MSE', train_mse, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                if epoch < 10:
                    scheduler.step()

                if valid_mse < best_mse:
                    best_mse = valid_mse
                    torch.save(protocol_model.state_dict(), f'checkpoints/mlp_checkpoint{i}.pt')

    protocol_model.load_state_dict(torch.load(f'checkpoints/mlp_checkpoint{i}.pt'))
    mae, mse, r2, pearson_score = test(protocol_model, test_loader)
    print(f'Test MAE: {mae:.3f}')
    print(f'Test MSE: {mse:.3f}')
    print(f'Test r2 score: {r2:.3f}')
    print(f'Test pearson score: {pearson_score:.3f}')
    mae_list.append(mae)
    mse_list.append(mse)
    r2_list.append(r2)
    pearson_list.append(pearson_score)


mae_arr = np.array(mae_list)
mse_arr = np.array(mse_list)
rmse_arr = np.sqrt(mse_arr)
r2_arr = np.array(r2_list)
pearson_list = np.array(pearson_list)
print(f"MAE: {mae_arr.mean():.3f} ({mae_arr.std():.3f})")
print(f"MSE: {mse_arr.mean():.3f} ({mse_arr.std():.3f})")
print(f"RMSE: {rmse_arr.mean():.3f} ({rmse_arr.std():.3f})")
print(f"R2: {r2_arr.mean():.3f} ({r2_arr.std():.3f})")
print(f"Pearson: {pearson_list.mean():.3f} ({pearson_list.std():.3f})")



# %%
