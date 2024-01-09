from abc import ABC, abstractmethod
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import csv
import copy
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from feature.graph_featurizer import GraphFeaturizer

import pandas as pd
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import re
import numpy as np
import random
from rdkit.Chem.rdchem import HybridizationType, BondType
from torch_geometric.data import Data

def valid_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi.strip("\n"))
        #print(mol)
        if mol is not None:
            return True
        else:
            return False
    except:
        return False

BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}

class BaseDataset(Dataset): #, ABC):
    def __init__(self, config):
        super(BaseDataset, self).__init__()
        self.config = config
        self._load_data()
        
    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.smiles)

class MolDataset(BaseDataset):
    """
    This function processes the dataset for machine learning tasks.
    
    Parameters:
        data_path (str): The path to the source data file. Supported formats include pkl, csv, and txt. The data will be read using pandas.
        
        config (dict): A dictionary containing various settings and configurations for data processing.
        
        split (str): Specifies the type of dataset to be returned. Options are 'train', 'valid', and 'test'.
        
        tokenizer_org (Tokenizer, optional): A tokenizer for processing the source data. If None, a default tokenizer will be used.
        
        tokenizer_label (Tokenizer, optional): A tokenizer for processing the label data. If None, the tokenizer_org will be used.
        
        task (dict, optional): Task in this project, Specifies the modality of the input and output data.
        
        transform (callable, optional): A function/transform to apply to image data for normalization or data augmentation.
    
    Returns:
        processed_data (Dataset): A processed dataset ready for training or evaluation.
    """
    def __init__(self, data_path, config, split=None, tokenizer_org = None, tokenizer_label = None, task = None, transform=None):
        if data_path.endswith('.pkl'):
            self.data = pd.read_pickle(data_path)
        elif data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith('.txt'):
            self.data = pd.read_table(data_path)
        else:
            raise ValueError(f'Unsupported file extension in: {data_path}')
            
        self.split = split
      
        self.tokenizer_org = tokenizer_org
        if(tokenizer_label == None):
            self.tokenizer_label = self.tokenizer_org
        else:
            self.tokenizer_label = tokenizer_label
        #graph_feature
        graph2d_featurizer_config = { 
            'name' : 'ogb'
        }
        self.task = task
        
        self.graph2d_featurizer = GraphFeaturizer(graph2d_featurizer_config)
        if(self.split == 'train'):
            self.transform = transforms.Compose([
                #transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                # transforms.RandomErasing(p=0.3,scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        super(MolDataset, self).__init__(config)
                                               #for bond in mol.GetBonds())))
    def _load_data(self):
        self.smiles = []
        self.isosmiles = []
        self.captions = []
        self.cids = []
        self.image2d = []
        self.graph2d = []
        self.iupac = []
        self.inchi = []
        self.selfies = []
        self.input_modal = []
        self.output_modal = []
        self.org_xlogp = []
        self.target_xlogp = []
        self.org_polararea = []
        self.target_polararea = []
        self.target_SMILES = []
        self.target_IUPAC = []
        self.scaffold = []

        self.input_modal = self.task['input_modal']
        self.output_modal = self.task['output_modal']
        self.modality = self.input_modal + self.output_modal
        #print(self.input_modal)
        for _, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            smiles = row["SMILES"]
            #print(smiles)
            if valid_smiles(smiles):
                #print(smiles)
                if "IUPAC" in self.modality or "SELFIES" in self.modality:
                    if pd.isnull(row["iupacname"]) or pd.isnull(row["SELFIES"]):
                        continue
                #smiles always append
                self.smiles.append(row["SMILES"])
                if "caption" in self.modality:
                    self.captions.append(row["description"])
                if "IUPAC" in self.modality:
                    self.iupac.append(row["iupacname"])
                if "InChI" in self.modality:
                    self.inchi.append(row["inchi"])
                if "SELFIES" in self.modality:
                    self.selfies.append(row["SELFIES"])

                if "image" in self.modality:
                    img_file2d = self.task['image_path']+'/CID_'+str(row["CID"])+'.png'
                    #img_file2d = '../../'+ row["image_path"]
                    img2d = Image.open(img_file2d).convert('RGB')
                    img2d = self.transform(img2d)  
                    self.image2d.append(img2d)
                if "graph" in self.modality:
                    graph2d = self.graph2d_featurizer(smiles)
                    self.graph2d.append(graph2d)
                if(self.task['name'] in ['molopt2smi','molopt2IUPAC']):
                    self.target_property = self.task['target_property']
                    self.scaffold.append(row["scaffold"])
                    if(self.target_property == "xlogp"):
                        self.org_xlogp.append(int(row["xlogp"]))
                        self.target_xlogp.append(int(row["target_xlogp"]))
                    else:
                        self.org_polararea.append(int(row["polararea"]))
                        self.target_polararea.append(int(row["target_polararea"]))
                if(self.task['name'] == 'molopt2smi'):
                    self.target_SMILES.append(row["target_SMILES"])
                if(self.task['name'] == 'molretri'):
                    self.cids.append(row['CID'])
            else:
                pass
    
    def __getitem__(self, i):
        mol_data = {}
        #input_modal_data
        if(self.task['name'] == 'molretri'):
            mol_data["cid"] = torch.tensor(self.cids[i])
        if "image" in self.input_modal:
            mol_data["image"] = self.image2d[i]
        if('SMILES' in self.input_modal):
            smiles = self.tokenizer_org(
                self.smiles[i],
                return_tensors="pt"
            )
            mol_data['SMILES'] = smiles
        if('caption' in self.input_modal):
            caption = self.tokenizer_org(
                self.captions[i],
                return_tensors="pt"
            )
            mol_data['caption'] = caption
        if('IUPAC' in self.input_modal):
            iupac = self.tokenizer_org(
                str(self.iupac[i]),
                return_tensors="pt"
            )
            mol_data['IUPAC'] = iupac
        if('InChI' in self.input_modal):
            inchi = self.tokenizer_org(
                self.inchi[i],
                return_tensors="pt",
            )
            mol_data['InChI'] = inchi
        if('SELFIES' in self.input_modal):
            selfies = self.tokenizer_org(
                self.selfies[i],
                return_tensors="pt",
            )
            mol_data['SELFIES'] = selfies
            
        if('graph' in self.input_modal):
            mol_data['graph'] = self.graph2d[i]

        if(self.task['name'] in ['molopt2smi','molopt2IUPAC']):
            mol_data['scaffold'] = self.tokenizer_label(
                self.scaffold[i],
                return_tensors="pt",
            )
            if(self.target_property == "xlogp"):
                mol_data['org_property'] = torch.tensor(self.org_xlogp[i], dtype=torch.float32)
                mol_data['target_property'] = torch.tensor(self.target_xlogp[i], dtype=torch.float32)
            else:
                mol_data['org_property'] = torch.tensor(self.org_polararea[i], dtype=torch.float32)
                mol_data['target_property'] = torch.tensor(self.target_polararea[i], dtype=torch.float32)
        #output_modal_label_data
        if('SMILES' in self.output_modal):
            if(self.task['name'] == 'molopt2smi'):
                smiles_labels = self.tokenizer_label(
                self.target_SMILES[i],
                return_tensors="pt",
                )
            else:
                smiles_labels = self.tokenizer_label(
                    self.smiles[i],
                    return_tensors="pt",
                )     
            mol_data['SMILES_labels'] = smiles_labels
            
        if(self.task['name'] == 'molopt2smi') and (self.split == 'valid' or self.split == 'test'):
            mol_data['SMILES_truth'] = self.target_SMILES[i]
        elif('SMILES' in self.modality or 'image' in self.modality or 'graph' in self.modality) and (self.split == 'valid' or self.split == 'test'):
            mol_data['SMILES_truth'] = self.smiles[i]
            
        if('caption' in self.output_modal):
            caption_labels = self.tokenizer_label(
                self.captions[i],
                return_tensors="pt",
            )
            mol_data['caption_labels'] = caption_labels
        if('caption' in self.modality) and (self.split == 'valid' or self.split == 'test'):
            mol_data['caption_truth'] = self.captions[i]
            
        if('IUPAC' in self.output_modal):
            if(self.task['name'] == 'molopt2IUPAC'):
                IUPAC_labels = self.tokenizer_label(
                    str(self.target_IUPAC[i]),
                    return_tensors="pt",
                )
            else:
                IUPAC_labels = self.tokenizer_label(
                    self.iupac[i],
                    return_tensors="pt",
                )
            mol_data['IUPAC_labels'] = IUPAC_labels
            
        if(self.task['name'] == 'molopt2IUPAC') and (self.split == 'valid' or self.split == 'test'):
            mol_data['IUPAC_truth'] = self.target_IUPAC[i] 
        elif('IUPAC' in self.modality) and (self.split == 'valid' or self.split == 'test'):
            mol_data['IUPAC_truth'] = self.iupac[i]  

        if('InChI' in self.modality) and (self.split == 'valid' or self.split == 'test'):
            mol_data['InChI_truth'] = self.inchi[i]     

        if('SELFIES' in self.modality) and (self.split == 'valid' or self.split == 'test'):
            mol_data['SELFIES_truth'] = self.selfies[i]
            
        return mol_data
