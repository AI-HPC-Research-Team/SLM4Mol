import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

class GPT2(nn.Module):
    def __init__(self, config = None):
        super(GPT2, self).__init__()
        if(config == None):
            config = {
                "name":"gpt2",
                "ckpt":"/workspace/lpf/CLM-insights/ckpts/text_ckpts/gpt2"
            }
        self.main_model = GPT2LMHeadModel.from_pretrained(config['ckpt'])
        self.tokenizer = GPT2Tokenizer.from_pretrained(config['ckpt'], model_max_length=512, padding_side='left')
        if "stop_grad" in config:
            for k, v in self.main_model.named_parameters():
                v.requires_grad = False
        self.hidden_size = self.main_model.config.hidden_size
        #self.fc = nn.Linear(768, 1024)  
        #self.output_dim = 1024
        
        

    def forward(self, encoder_outputs, attention_mask, decoder_attention_mask, labels):
        h = encoder_outputs.last_hidden_state

        h_attention_mask = attention_mask
        
        #targets = labels['input_ids'].masked_fill(labels['input_ids'] == self.tokenizer.pad_token_id, -100)
        #print(labels)
        targets = labels
        inputs_embeds = self.main_model.transformer.wte(labels)
        #print(f"h.shape:{h.shape}")
        #print(f"inputs_embeds.shape:{inputs_embeds.shape}")
        inputs_embeds = torch.cat([h,inputs_embeds], dim=1)
        try:
            labels_attention_mask = labels['attention_mask']
        except:
            labels_attention_mask = torch.ones_like(labels)
        attention_mask = torch.cat([
            h_attention_mask,
            labels_attention_mask.to(h.device)], dim=1)  

        empty_targets = (
            torch.ones(h_attention_mask.size(), dtype=torch.long).to(h.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        
        outputs = self.main_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels = targets,
        )
        return outputs['loss']
    def encode(self, mol):
        inputs_embeds = self.main_model.transformer.wte(mol['input_ids'])
        return inputs_embeds
    
    def decode(self, encoder_outputs, encoder_attention_mask, num_beams, max_length, min_length = 5):
        h = encoder_outputs.last_hidden_state
        
        if encoder_attention_mask.dtype != torch.float:
            encoder_attention_mask = encoder_attention_mask.float()
            
        outputs = self.main_model.generate(
            encoder_hidden_states= h,
            encoder_attention_mask=encoder_attention_mask, 
            num_beams = num_beams,
            max_length= max_length,
            min_length = min_length,
            early_stopping=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

