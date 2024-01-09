import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, RobertaForCausalLM, AutoConfig, RobertaModel
import torch

class Roberta(nn.Module):
    def __init__(self, config = None):
        super(Roberta, self).__init__()
        if(config == None):
            config = {
                "name":"roberta",
                "ckpt":"/workspace/lpf/CLM-insights/ckpts/text_ckpts/roberta"
            }

        self.encoder = RobertaModel.from_pretrained(config["ckpt"])
        self.dropout = nn.Dropout(0.1)
        self.output_dim = 768
        
    def forward(self, drug):
        h = self.encoder(**drug).last_hidden_state
        return h
    
    def encode(self, text):
        h = self.encoder(**text).last_hidden_state
        #h = h.transpose(1, 2) 
        #h = self.pool(h, h.shape[2]).squeeze(2)
        return h

class ChemBERTa(nn.Module):
    def __init__(self, config = None):
        super(ChemBERTa, self).__init__()
        if(config == None):
            config = {
                "name":"chemberta",
                "ckpt":"/workspace/lpf/CLM-insights/ckpts/text_ckpts/chemberta",
            }
        self.encoder = RobertaModel.from_pretrained(config["ckpt"])
        self.dropout = nn.Dropout(0.1)
        self.output_dim = 768

        
    def forward(self, drug):
        h = self.encoder(**drug).last_hidden_state
        return h
    
    def encode(self, text):
        """
        for key in text.keys():
            print(text[key].shape)
        """
        h = self.encoder(
            input_ids=text['input_ids'], 
            attention_mask=text['attention_mask']
        ).last_hidden_state
        return h
