import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPTNeoForCausalLM, AutoTokenizer
import torch

import inspect

class ChemGPT(nn.Module):
    def __init__(self, config = None):
        super(ChemGPT, self).__init__()
        if(config == None):
            config = {
                "name":"chemgpt",
                "ckpt":"/workspace/lpf/CLM-insights/ckpts/text_ckpts/chemgpt"
            }
        self.main_model = GPTNeoForCausalLM.from_pretrained(config['ckpt'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['ckpt'], model_max_length=512)
        if "stop_grad" in config:
            for k, v in self.main_model.named_parameters():
                v.requires_grad = False
        self.hidden_size = self.main_model.config.hidden_size
        self.fc = nn.Linear(768, 128)  
        #self.output_dim = 1024
        
        

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels):
        
        #h = encoder_outputs.last_hidden_state
        #h_attention_mask = attention_mask

        #targets = labels['input_ids'].masked_fill(labels['input_ids'] == self.tokenizer.pad_token_id, -100)
        #print(labels)
        #print(f"input_ids : {input_ids.shape}")
        #print(f"attention_mask : {attention_mask.shape}")
        #print(f"labels : {labels.shape}")
        targets = labels
        #inputs_embeds = self.main_model.transformer(labels)[0]
        #print(f"h.shape:{h.shape}")
        #print(f"inputs_embeds.shape:{inputs_embeds.shape}")
        input_ids = torch.cat([input_ids,targets], dim=1)
        empty_targets = (
            torch.ones(attention_mask.size(), dtype=torch.long).to(input_ids.device).fill_(-100)
        )
        labels_attention_mask = torch.ones_like(labels)
        attention_mask = torch.cat([
            attention_mask,
            labels_attention_mask.to(input_ids.device)], dim=1)  

        targets = torch.cat([empty_targets, targets], dim=1)
        """
        outputs = self.main_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels = targets,
        )
        """
        #print(f"input_ids : {input_ids.shape}")
        #print(f"attention_mask : {attention_mask.shape}")
        #print(f"labels : {targets.shape}")
        outputs = self.main_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            labels = targets,
        )
        return outputs['loss']
    def encode(self, mol):
        inputs_embeds = self.main_model.transformer(**mol)[0]
        return inputs_embeds
    
    def decode(self, input_ids, attention_mask, num_beams, max_length, min_length = 5):
        #model_args = set(inspect.signature(self.main_model.prepare_inputs_for_generation).parameters)
        #print(f"model_args1: {model_args}")
        #model_args |= set(inspect.signature(self.main_model.forward).parameters)
        #print(f"model_args2: {model_args}")
        #h = encoder_outputs.last_hidden_state
        outputs = self.main_model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask, 
            #do_sample=True,
            num_beams = num_beams,
            max_length= max_length,
            min_length = min_length
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

