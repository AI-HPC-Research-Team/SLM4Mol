import os
import numpy as np
import random
import torch

import datetime

import torch
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence

import logging
logger = logging.getLogger(__name__)

    
def ToDevice(obj, device):
    if isinstance(obj, (str)):
        return obj
    if isinstance(obj, (int)):
        return obj
    if isinstance(obj, dict):
        for k in obj:
            obj[k] = ToDevice(obj[k], device)
        return obj
    elif isinstance(obj, tuple):
        # Convert tuple to list, modify the list, then convert back to tuple
        obj = list(obj)
        for i in range(len(obj)):
            obj[i] = ToDevice(obj[i], device)
        return tuple(obj)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = ToDevice(obj[i], device)
        return obj
    else:
        return obj.to(device)
    
def print_model_info(model, level=0, prefix=''):
    total_params = 0
    trainable_params = 0

    for name, module in model.named_children():
        total_params_module = sum(p.numel() for p in module.parameters())
        trainable_params_module = sum(p.numel() for p in module.parameters() if p.requires_grad)

        total_params += total_params_module
        trainable_params += trainable_params_module

        print(f"{prefix}Module: {name} | Total parameters: {total_params_module} | Trainable parameters: {trainable_params_module}")

        if level > 0:
            print_model_info(module, level=level-1, prefix=prefix + '  ')

    if prefix == '':
        print(f"Total parameters: {total_params} | Trainable parameters: {trainable_params} | Trainable ratio: {trainable_params / total_params:.2%}")


def custom_collate_fn(batch):
    collated_batch = {}
    elem_keys = batch[0].keys()
    max_length = 511  # max length

    for key in elem_keys:
        if key in ['SMILES_labels','caption_labels','IUPAC_labels','InChI_labels','SELFIES_labels', 'SMILES','caption','IUPAC','InChI','SELFIES','text','scaffold']:
            input_ids_tensors = [elem[key].input_ids for elem in batch]
            input_ids_tensors = [tensor.squeeze(0) for tensor in input_ids_tensors]
            padded_input_ids = pad_sequence(input_ids_tensors, batch_first=True)
            
            # max length 512
            if padded_input_ids.size(1) > max_length:
                padded_input_ids = padded_input_ids[:, :max_length]

            attention_mask_tensors = [elem[key].attention_mask for elem in batch]
            attention_mask_tensors = [tensor.squeeze(0) for tensor in attention_mask_tensors]
            padded_attention_mask = pad_sequence(attention_mask_tensors, batch_first=True)

            # max length 512
            if padded_attention_mask.size(1) > max_length:
                padded_attention_mask = padded_attention_mask[:, :max_length]

            collated_batch[key] = {
                'input_ids': padded_input_ids,
                'attention_mask': padded_attention_mask
            }
        elif key in ['SMILES_truth','caption_truth','IUPAC_truth','InChI_truth','SELFIES_truth']:
            collated_batch[key] = [item[key] for item in batch]
        elif(key in ['graph']):
            molecule_batch = Batch.from_data_list([elem[key] for elem in batch])
            collated_batch[key] = molecule_batch
        elif(key in ['org_property','target_property','cid','id']):
            padded_data = torch.stack([elem[key] for elem in batch])
            padded_data = padded_data.squeeze(0)
            collated_batch[key] = padded_data
        else:
            #print(key)
            padded_data = torch.stack([elem[key] for elem in batch])
            padded_data = padded_data.squeeze(1)
            collated_batch[key] = padded_data
              

    return collated_batch