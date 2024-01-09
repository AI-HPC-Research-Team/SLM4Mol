import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BartTokenizer, BartModel, BartForConditionalGeneration
import torch

class BART(nn.Module):
    def __init__(self, config = None):
        super(BART, self).__init__()
        if(config == None):
            config = {
                "name":"bart",
                "ckpt":"/workspace/lpf/CLM-insights/ckpts/text_ckpts/bart"
            }
        self.main_model = BartForConditionalGeneration.from_pretrained(config['ckpt'])
        self.tokenizer = BartTokenizer.from_pretrained(config['ckpt'], model_max_length=512)
        if "stop_grad" in config:
            for k, v in self.main_model.named_parameters():
                v.requires_grad = False
        self.hidden_size = self.main_model.config.hidden_size
        self.output_dim = self.hidden_size

    def forward(self, encoder_outputs, attention_mask, decoder_attention_mask, labels):
        return self.main_model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        ).loss
    
    def encode(self, mol):
        h = self.main_model.model(**mol).encoder_last_hidden_state
        return h

    def decode(self, encoder_outputs, encoder_attention_mask, num_beams, max_length):
        outputs = self.main_model.generate(
            encoder_outputs = encoder_outputs,  
            attention_mask = encoder_attention_mask, 
            num_beams=num_beams,
            max_length=max_length,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def encode_mol(self, mol):
        return self.encode(mol)

    def encode_text(self, text):
        return self.main_model.encoder(**text)
    
