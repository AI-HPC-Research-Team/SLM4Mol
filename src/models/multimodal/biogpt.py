import torch
import torch.nn as nn

from transformers import BioGptForCausalLM, BioGptTokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
#from models.base_models import MolEncoder, TextEncoder

class BioGPT(nn.Module):
    def __init__(self, config = None):
        super(BioGPT, self).__init__()
        if(config == None):
            config = {
                "name":"biogpt",
                "ckpt":"/workspace/lpf/CLM-insights/ckpts/text_ckpts/biogpt"
            }
        self.main_model = BioGptForCausalLM.from_pretrained(config['ckpt'])
        self.embed_scale = self.main_model.biogpt.embed_scale
        self.tokenizer = BioGptTokenizer.from_pretrained(config['ckpt'], model_max_length=512)
        if "stop_grad" in config:
            for k, v in self.main_model.named_parameters():
                v.requires_grad = False
        self.hidden_size = self.main_model.config.hidden_size
        self.fc = nn.Linear(768, 1024)
        
        #self.output_dim = 1024

    def forward_x(self, input_ids, attention_mask, decoder_attention_mask, labels):
        targets = labels
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
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels = targets,
        )
        return outputs['loss']
    
    def forward(self, encoder_outputs, attention_mask, decoder_attention_mask, labels):
        encoder_outputs = encoder_outputs.last_hidden_state
        if(encoder_outputs.shape[2]==768):
            h = self.fc(encoder_outputs)
        else:
            h = encoder_outputs
        h_attention_mask = attention_mask
        
        #targets = labels['input_ids'].masked_fill(labels['input_ids'] == self.tokenizer.pad_token_id, -100)
        targets = labels
        inputs_embeds = self.main_model.biogpt.embed_tokens(labels)*self.main_model.biogpt.embed_scale
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
        inputs_embeds = self.main_model.biogpt.embed_tokens(mol['input_ids'])*self.main_model.biogpt.embed_scale
        return inputs_embeds

    def decode(self, encoder_outputs, encoder_attention_mask, num_beams, max_length, min_length = 5):
        h = encoder_outputs.last_hidden_state
        if(h.shape[2]==768):
            h = self.fc(h)
        
        if encoder_attention_mask.dtype != torch.float:
            encoder_attention_mask = encoder_attention_mask.float()
            
        outputs = self.main_model.generate(
            inputs_embeds = h,
            attention_mask = encoder_attention_mask,
            num_beams = num_beams,
            max_length= max_length,
            min_length = min_length,
            early_stopping=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    def decode_input_ids(self, input_ids, attention_mask, num_beams, max_length, min_length = 5):
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