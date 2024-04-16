from multiprocessing import pool
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.parameter import Parameter


class Protocol_Embedding_Regression(nn.Sequential):
    def __init__(self, output_dim):
        super(Protocol_Embedding_Regression, self).__init__()    
        self.input_dim = 768  
        self.output_dim = output_dim 
        self.fc = nn.Linear(self.input_dim*2, output_dim)
        self.f = F.relu

    def forward(self, inclusion_emb, inclusion_mask, exclusion_emb, exclusion_mask):
        inclusion_vec = torch.mean(inclusion_emb * inclusion_mask.unsqueeze(-1), dim=1)
        exclusion_vec = torch.mean(exclusion_emb * exclusion_mask.unsqueeze(-1), dim=1)
        
        protocol_mat = torch.cat([inclusion_vec, exclusion_vec], 1)
        output = self.f(self.fc(protocol_mat))
        return output 

    @property
    def embedding_size(self):
        return self.output_dim 


class Protocol_Attention_Regression(nn.Module):
    def __init__(self, sentence_embedding_dim, linear_output_dim, transformer_encoder_layers=2, num_heads=6, dropout=0.1, pooling_method='cls'):
        super(Protocol_Attention_Regression, self).__init__()

        # Validate pooling method
        if pooling_method not in ["mean", "max", "cls"]:
            print(f"Invalid pooling method: {pooling_method}. Using 'cls' pooling method.")
            pooling_method = "cls"
        self.pooling_method = pooling_method

        self.cls_token = Parameter(torch.rand(1, 1, sentence_embedding_dim))

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=sentence_embedding_dim, nhead=num_heads, dropout=dropout, batch_first=True, dim_feedforward=sentence_embedding_dim)
        layer_norm = nn.LayerNorm(sentence_embedding_dim)
        
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=transformer_encoder_layers, norm=layer_norm)
        
        # self.linear1 = nn.Linear(4*sentence_embedding_dim+4, sentence_embedding_dim)
        self.linear1 = nn.Linear(3*sentence_embedding_dim+4, sentence_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(sentence_embedding_dim, linear_output_dim)
        

    def forward(self, inclusion_emb, inclusion_mask, exclusion_emb, exclusion_mask, drug_emb, disease_emb, phase_emb):
        # Expand cls_token to match the batch size
        # inclu_emb = torch.cat((self.cls_token.expand(inclusion_emb.shape[0], -1, -1), inclusion_emb), dim=1)
        # inclu_mask = torch.cat((torch.ones(inclusion_emb.shape[0], 1, dtype=torch.bool, device=inclusion_emb.device), inclusion_mask), dim=1)
        # inclu_encoded = self.transformer_encoder(inclu_emb, src_key_padding_mask=inclu_mask)

        # exclu_emb = torch.cat((self.cls_token.expand(exclusion_emb.shape[0], -1, -1), exclusion_emb), dim=1)
        # exclu_mask = torch.cat((torch.ones(exclusion_emb.shape[0], 1, dtype=torch.bool, device=exclusion_emb.device), exclusion_mask), dim=1)
        # exclu_encoded = self.transformer_encoder(exclu_emb, src_key_padding_mask=exclu_mask)

        criteria_emb = torch.cat((inclusion_emb, exclusion_emb), dim=1)
        criteria_emb = torch.cat((self.cls_token.expand(criteria_emb.shape[0], -1, -1), criteria_emb), dim=1)

        criteria_mask = torch.cat((inclusion_mask, exclusion_mask), dim=1)
        criteria_mask = torch.cat((torch.ones(criteria_emb.shape[0], 1, dtype=torch.bool, device=criteria_emb.device), criteria_mask), dim=1)
        criteria_encoded = self.transformer_encoder(criteria_emb, src_key_padding_mask=criteria_mask)

        # Adjust pooling method handling
        # if self.pooling_method == "cls":
        #     inclu_pooled_emb = inclu_encoded[:, 0, :]
        #     exclu_pooled_emb = exclu_encoded[:, 0, :]
        # elif self.pooling_method == "max":
        #     inclu_pooled_emb = torch.max(inclu_encoded, dim=1)[0]
        #     exclu_pooled_emb = torch.max(exclu_encoded, dim=1)[0]
        # elif self.pooling_method == "mean":
        #     inclu_pooled_emb = torch.mean(inclu_encoded, dim=1)
        #     exclu_pooled_emb = torch.mean(exclu_encoded, dim=1)
        # else:
        #     raise ValueError("Invalid pooling method")

        if self.pooling_method == "cls":
            pooled_emb = criteria_encoded[:, 0, :]
        elif self.pooling_method == "max":
            pooled_emb = torch.max(criteria_encoded, dim=1)[0]
        elif self.pooling_method == "mean":
            pooled_emb = torch.mean(criteria_encoded, dim=1)
        else:
            raise ValueError("Invalid pooling method")

        protocol_emb = torch.cat((pooled_emb, drug_emb, disease_emb, phase_emb), dim=1)
        # protocol_emb = torch.cat((inclu_pooled_emb, exclu_pooled_emb, drug_emb, disease_emb, phase_emb), dim=1)
        output = self.linear1(protocol_emb)
        output = self.dropout(output)
        output = self.relu(output)
        output = self.linear2(output)
        
        return output
    
class Protocol_Attention_Regression_With_Phases(nn.Module):
    def __init__(self, sentence_embedding_dim, linear_output_dim, transformer_encoder_layers=2, num_heads=6, dropout=0.1, pooling_method='cls'):
        super(Protocol_Attention_Regression_With_Phases, self).__init__()

        # Validate pooling method
        if pooling_method not in ["mean", "max", "cls"]:
            print(f"Invalid pooling method: {pooling_method}. Using 'cls' pooling method.")
            pooling_method = "cls"
        self.pooling_method = pooling_method

        self.cls_token = Parameter(torch.rand(1, 1, sentence_embedding_dim))

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=sentence_embedding_dim, nhead=num_heads, dropout=dropout, batch_first=True, dim_feedforward=sentence_embedding_dim)
        layer_norm = nn.LayerNorm(sentence_embedding_dim)
        
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=transformer_encoder_layers, norm=layer_norm)
        
        # self.linear1 = nn.Linear(4*sentence_embedding_dim+4, sentence_embedding_dim)
        self.linear1 = nn.Linear(3*sentence_embedding_dim, sentence_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(sentence_embedding_dim, linear_output_dim)
        

    def forward(self, inclusion_emb, inclusion_mask, exclusion_emb, exclusion_mask, drug_emb, disease_emb):
        # Expand cls_token to match the batch size
        # inclu_emb = torch.cat((self.cls_token.expand(inclusion_emb.shape[0], -1, -1), inclusion_emb), dim=1)
        # inclu_mask = torch.cat((torch.ones(inclusion_emb.shape[0], 1, dtype=torch.bool, device=inclusion_emb.device), inclusion_mask), dim=1)
        # inclu_encoded = self.transformer_encoder(inclu_emb, src_key_padding_mask=inclu_mask)

        # exclu_emb = torch.cat((self.cls_token.expand(exclusion_emb.shape[0], -1, -1), exclusion_emb), dim=1)
        # exclu_mask = torch.cat((torch.ones(exclusion_emb.shape[0], 1, dtype=torch.bool, device=exclusion_emb.device), exclusion_mask), dim=1)
        # exclu_encoded = self.transformer_encoder(exclu_emb, src_key_padding_mask=exclu_mask)

        criteria_emb = torch.cat((inclusion_emb, exclusion_emb), dim=1)
        criteria_emb = torch.cat((self.cls_token.expand(criteria_emb.shape[0], -1, -1), criteria_emb), dim=1)

        criteria_mask = torch.cat((inclusion_mask, exclusion_mask), dim=1)
        criteria_mask = torch.cat((torch.ones(criteria_emb.shape[0], 1, dtype=torch.bool, device=criteria_emb.device), criteria_mask), dim=1)
        criteria_encoded = self.transformer_encoder(criteria_emb, src_key_padding_mask=criteria_mask)

        # Adjust pooling method handling
        # if self.pooling_method == "cls":
        #     inclu_pooled_emb = inclu_encoded[:, 0, :]
        #     exclu_pooled_emb = exclu_encoded[:, 0, :]
        # elif self.pooling_method == "max":
        #     inclu_pooled_emb = torch.max(inclu_encoded, dim=1)[0]
        #     exclu_pooled_emb = torch.max(exclu_encoded, dim=1)[0]
        # elif self.pooling_method == "mean":
        #     inclu_pooled_emb = torch.mean(inclu_encoded, dim=1)
        #     exclu_pooled_emb = torch.mean(exclu_encoded, dim=1)
        # else:
        #     raise ValueError("Invalid pooling method")

        if self.pooling_method == "cls":
            pooled_emb = criteria_encoded[:, 0, :]
        elif self.pooling_method == "max":
            pooled_emb = torch.max(criteria_encoded, dim=1)[0]
        elif self.pooling_method == "mean":
            pooled_emb = torch.mean(criteria_encoded, dim=1)
        else:
            raise ValueError("Invalid pooling method")

        protocol_emb = torch.cat((pooled_emb, drug_emb, disease_emb), dim=1)
        # protocol_emb = torch.cat((inclu_pooled_emb, exclu_pooled_emb, drug_emb, disease_emb, phase_emb), dim=1)
        output = self.linear1(protocol_emb)
        output = self.dropout(output)
        output = self.relu(output)
        output = self.linear2(output)
        
        return output
    
