import torch
import random
import numpy as np
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

# splitter function

class MolasubDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)


def generate_scaffold(smiles, include_chirality=False):
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=include_chirality)
        return scaffold
    except ValueError:
        print(f"Failed to generate scaffold for SMILES: {smiles}")
        return None  # or return the original SMILES or any other default value


def scaffold_split(dataset, tokenizer, smiles_list, text_modal=None, text_list=None, task_idx=None, null_value=0,
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                   return_smiles=True):
 
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

        
    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    if not return_smiles:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_data = []
        valid_data = []
        test_data = []
        #print(text_list)
        if text_list is not None:
            ##print("XXXX")
            train_smiles = [tokenizer(smiles_list[i][1], return_tensors="pt") for i in train_idx]
            valid_smiles = [tokenizer(smiles_list[i][1], return_tensors="pt") for i in valid_idx]
            test_smiles = [tokenizer(smiles_list[i][1], return_tensors="pt")for i in test_idx]
            for idx, smiles in enumerate(train_smiles):
                text_input = text_list[train_idx[idx]]
                text_input = str(text_input) if text_input is not None else None
                if text_input is not None:  
                    data = {
                        #'graph': train_dataset[idx],
                        #'SMILES': smiles,
                        'id': torch.tensor(train_idx[idx]),
                        text_modal: tokenizer(text_input, return_tensors="pt"),
                        'label' : train_dataset[idx].y
                    }
                    train_data.append(data)

            for idx, smiles in enumerate(valid_smiles):
                text_input = text_list[valid_idx[idx]]
                text_input = str(text_input) if text_input is not None else None
                if text_input is not None:  
                    data = {
                        #'graph': valid_dataset[idx],
                        #'SMILES': smiles,
                        'id': torch.tensor(valid_idx[idx]),
                        text_modal: tokenizer(text_input, return_tensors="pt"),
                        'label' : valid_dataset[idx].y
                    }
                    valid_data.append(data)

            for idx, smiles in enumerate(test_smiles):
                text_input = text_list[test_idx[idx]]
                text_input = str(text_input) if text_input is not None else None
                if text_input is not None:  
                    data = {
                        #'graph': test_dataset[idx],
                        #'SMILES': smiles,
                        'id': torch.tensor(test_idx[idx]),
                        text_modal: tokenizer(text_input, return_tensors="pt"),
                        'label' : test_dataset[idx].y
                    }
                    test_data.append(data)
        elif text_modal == 'SMILES':
            train_smiles = [tokenizer(smiles_list[i][1], return_tensors="pt") for i in train_idx]
            valid_smiles = [tokenizer(smiles_list[i][1], return_tensors="pt") for i in valid_idx]
            test_smiles = [tokenizer(smiles_list[i][1], return_tensors="pt")for i in test_idx]
            for i in range(len(train_smiles)):
                #print(train_smiles[i])
                data = {
                    #'graph' : train_dataset[i],
                    'id': torch.tensor(train_idx[i]),
                    'SMILES': train_smiles[i],
                    'label' : train_dataset[i].y
                }
                train_data.append(data)
            for i in range(len(valid_smiles)):
                data = {
                    #'graph' : valid_dataset[i],
                    'id': torch.tensor(valid_idx[i]),
                    'SMILES': valid_smiles[i],
                    'label' : valid_dataset[i].y
                }
                valid_data.append(data)
            for i in range(len(test_smiles)):
                data = {
                    #'graph' : test_dataset[i],
                    'id': torch.tensor(test_idx[i]),
                    'SMILES': test_smiles[i],
                    'label' : test_dataset[i].y
                }
                test_data.append(data)
        else:
            train_smiles = [smiles_list[i][1] for i in train_idx]
            valid_smiles = [smiles_list[i][1] for i in valid_idx]
            test_smiles = [smiles_list[i][1] for i in test_idx]
            for i in range(len(train_smiles)):
                #print(train_smiles[i])
                data = {
                    'id': torch.tensor(train_idx[i]),
                    'graph' : train_dataset[i],
                    'SMILES_truth': train_smiles[i],
                    'label' : train_dataset[i].y
                }
                train_data.append(data)
            for i in range(len(valid_smiles)):
                data = {
                    'id': torch.tensor(valid_idx[i]),
                    'graph' : valid_dataset[i],
                    'SMILES_truth': valid_smiles[i],
                    'label' : valid_dataset[i].y
                }
                valid_data.append(data)
            for i in range(len(test_smiles)):
                data = {
                    'id': torch.tensor(test_idx[i]),
                    'graph' : test_dataset[i],
                    'SMILES_truth': test_smiles[i],
                    'label' : test_dataset[i].y
                }
                test_data.append(data)
        train_dataset = MolasubDataset(train_data)
        valid_dataset = MolasubDataset(valid_data)
        test_dataset = MolasubDataset(test_data)
        return train_dataset, valid_dataset, test_dataset
        
def random_split(dataset, tokenizer, smiles_list, text_modal=None, text_list=None, task_idx=None, null_value=0,
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                   return_smiles=True):

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        non_null = y_task != null_value  # boolean array that correspond to non null values
        idx_array = np.where(non_null)[0]
        dataset = dataset[torch.tensor(idx_array)]  # examples containing non
        # null labels in the specified task_idx
    else:
        pass

    num_mols = len(dataset)
    #random.seed(seed)
    all_idx = list(range(num_mols))
    random.shuffle(all_idx)

    train_idx = all_idx[:int(frac_train * num_mols)]
    valid_idx = all_idx[int(frac_train * num_mols):int(frac_valid * num_mols)
                                                   + int(frac_train * num_mols)]
    test_idx = all_idx[int(frac_valid * num_mols) + int(frac_train * num_mols):]

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == num_mols

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    if not return_smiles:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_data = []
        valid_data = []
        test_data = []
        #print(text_list)
        if text_list is not None:
            ##print("XXXX")
            train_smiles = [tokenizer(smiles_list[i], return_tensors="pt") for i in train_idx]
            valid_smiles = [tokenizer(smiles_list[i], return_tensors="pt") for i in valid_idx]
            test_smiles = [tokenizer(smiles_list[i], return_tensors="pt")for i in test_idx]
            for idx, smiles in enumerate(train_smiles):
                text_input = text_list[train_idx[idx]]
                text_input = str(text_input) if text_input is not None else None
                if text_input is not None:  
                    data = {
                        #'graph': train_dataset[idx],
                        #'SMILES': smiles,
                        'id': torch.tensor(train_idx[idx]),
                        text_modal: tokenizer(text_input, return_tensors="pt"),
                        'label' : train_dataset[idx].y
                    }
                    train_data.append(data)

            for idx, smiles in enumerate(valid_smiles):
                text_input = text_list[valid_idx[idx]]
                text_input = str(text_input) if text_input is not None else None
                if text_input is not None:  
                    data = {
                        #'graph': valid_dataset[idx],
                        #'SMILES': smiles,
                        'id': torch.tensor(valid_idx[idx]),
                        text_modal: tokenizer(text_input, return_tensors="pt"),
                        'label' : valid_dataset[idx].y
                    }
                    valid_data.append(data)

            for idx, smiles in enumerate(test_smiles):
                text_input = text_list[test_idx[idx]]
                text_input = str(text_input) if text_input is not None else None
                if text_input is not None:  
                    data = {
                        #'graph': test_dataset[idx],
                        #'SMILES': smiles,
                        'id': torch.tensor(test_idx[idx]),
                        text_modal: tokenizer(text_input, return_tensors="pt"),
                        'label' : test_dataset[idx].y
                    }
                    test_data.append(data)
        elif text_modal == 'SMILES':
            train_smiles = [tokenizer(smiles_list[i], return_tensors="pt") for i in train_idx]
            valid_smiles = [tokenizer(smiles_list[i], return_tensors="pt") for i in valid_idx]
            test_smiles = [tokenizer(smiles_list[i], return_tensors="pt")for i in test_idx]
            for i in range(len(train_smiles)):
                #print(train_smiles[i])
                data = {
                    #'graph' : train_dataset[i],
                    'id': torch.tensor(train_idx[i]),
                    'SMILES': train_smiles[i],
                    'label' : train_dataset[i].y
                }
                train_data.append(data)
            for i in range(len(valid_smiles)):
                data = {
                    #'graph' : valid_dataset[i],
                    'id': torch.tensor(valid_idx[i]),
                    'SMILES': valid_smiles[i],
                    'label' : valid_dataset[i].y
                }
                valid_data.append(data)
            for i in range(len(test_smiles)):
                data = {
                    #'graph' : test_dataset[i],
                    'id': torch.tensor(test_idx[i]),
                    'SMILES': test_smiles[i],
                    'label' : test_dataset[i].y
                }
                test_data.append(data)
        else:
            train_smiles = [smiles_list[i] for i in train_idx]
            valid_smiles = [smiles_list[i] for i in valid_idx]
            test_smiles = [smiles_list[i] for i in test_idx]
            for i in range(len(train_smiles)):
                #print(train_smiles[i])
                data = {
                    'id': torch.tensor(train_idx[i]),
                    'graph' : train_dataset[i],
                    'SMILES_truth': train_smiles[i],
                    'label' : train_dataset[i].y
                }
                train_data.append(data)
            for i in range(len(valid_smiles)):
                data = {
                    'id': torch.tensor(valid_idx[i]),
                    'graph' : valid_dataset[i],
                    'SMILES_truth': valid_smiles[i],
                    'label' : valid_dataset[i].y
                }
                valid_data.append(data)
            for i in range(len(test_smiles)):
                data = {
                    'id': torch.tensor(test_idx[i]),
                    'graph' : test_dataset[i],
                    'SMILES_truth': test_smiles[i],
                    'label' : test_dataset[i].y
                }
                test_data.append(data)
        train_dataset = MolasubDataset(train_data)
        valid_dataset = MolasubDataset(valid_data)
        test_dataset = MolasubDataset(test_data)
        return train_dataset, valid_dataset, test_dataset

