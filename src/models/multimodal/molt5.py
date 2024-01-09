import torch
import torch.nn as nn

from transformers import T5Tokenizer,T5EncoderModel, T5ForConditionalGeneration, MT5ForConditionalGeneration, AutoTokenizer
#from models.base_models import MolEncoder, TextEncoder

class MolT5_large(nn.Module):
    def __init__(self, config = None):
        super(MolT5_large, self).__init__()
        if(config == None):
            config = {
                "name":"molt5-large",
                "ckpt":"/workspace/lpf/CLM-insights/ckpts/text_ckpts/molt5-large"
            }
        self.main_model = T5ForConditionalGeneration.from_pretrained(config['ckpt'])
        self.tokenizer = T5Tokenizer.from_pretrained(config['ckpt'], model_max_length=512)
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
        h = self.main_model.encoder(**mol).last_hidden_state
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

class MolT5(nn.Module):
    def __init__(self, config = None):
        super(MolT5, self).__init__()
        if(config == None):
            config = {
                "name":"molt5",
                "ckpt":"/workspace/lpf/CLM-insights/ckpts/text_ckpts/molt5-base"
            }
        self.main_model = T5ForConditionalGeneration.from_pretrained(config['ckpt'])
        self.tokenizer = T5Tokenizer.from_pretrained(config['ckpt'], model_max_length=512)
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
        h = self.main_model.encoder(**mol).last_hidden_state
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
    
    
class T5(nn.Module):
    def __init__(self, config = None):
        super(T5, self).__init__()
        if(config == None):
            config = {
                "name":"t5",
                "ckpt":"/workspace/lpf/CLM-insights/ckpts/text_ckpts/flan-t5-base"
            }
        self.main_model = T5ForConditionalGeneration.from_pretrained(config['ckpt'])
        self.tokenizer = T5Tokenizer.from_pretrained(config['ckpt'], model_max_length=512)
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
        h = self.main_model.encoder(**mol).last_hidden_state
        return h

    def decode(self, encoder_outputs, encoder_attention_mask, num_beams, max_length):
        outputs = self.main_model.generate(
            encoder_outputs = encoder_outputs,  
            attention_mask = encoder_attention_mask, 
            num_beams=num_beams,
            max_length=max_length,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    def get_attentions(self, encoder_outputs, attention_mask, decoder_attention_mask, labels, output_attentions):
        attentions =  self.main_model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_attentions = output_attentions
        ).decoder_attentions
        return attentions
        
    def encode_mol(self, mol):
        return self.encode(mol)

    def encode_text(self, text):
        return self.main_model.encoder(**text)
    
class T511(nn.Module):
    def __init__(self, config = None):
        super(T511, self).__init__()
        if(config == None):
            config = {
                "name":"t5",
                "ckpt":"/workspace/lpf/CLM-insights/ckpts/text_ckpts/t5-v1_1-base"
            }
        self.main_model = T5ForConditionalGeneration.from_pretrained(config['ckpt'])
        self.tokenizer = T5Tokenizer.from_pretrained(config['ckpt'], model_max_length=512)
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
        h = self.main_model.encoder(**mol).last_hidden_state
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

class MT5(nn.Module):
    def __init__(self, config = None):
        super(MT5, self).__init__()
        if(config == None):
            config = {
                "name":"t5",
                "ckpt":"/workspace/lpf/CLM-insights/ckpts/text_ckpts/flan-t5-base"
            }
        self.main_model = MT5ForConditionalGeneration.from_pretrained(config['ckpt'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['ckpt'], model_max_length=512)
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
        h = self.main_model.encoder(**mol).last_hidden_state
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