'''
Code from https://github.com/blender-nlp/MolT5
```bibtex
@article{edwards2022translation,
  title={Translation between Molecules and Natural Language},
  author={Edwards, Carl and Lai, Tuan and Ros, Kevin and Honke, Garrett and Ji, Heng},
  journal={arXiv preprint arXiv:2204.11817},
  year={2022}
}
```
'''

import argparse
import csv

import os.path as osp

import numpy as np

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def molfinger_evaluate(targets, preds, morgan_r=2, verbose=False):
    outputs = []
    bad_mols = 0

    for i in range(len(targets)):
        try:
            gt_smi = targets[i]
            ot_smi = preds[i]
            gt_m = Chem.MolFromSmiles(gt_smi)
            ot_m = Chem.MolFromSmiles(ot_smi)

            if ot_m == None: raise ValueError('Bad SMILES')
            outputs.append(('test', gt_m, ot_m))
        except:
            bad_mols += 1
            
    validity_score = len(outputs)/(len(outputs)+bad_mols)
    if verbose:
        print('validity:', validity_score)


    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []

    enum_list = outputs

    for i, (desc, gt_m, ot_m) in enumerate(enum_list):

        if i % 100 == 0:
            if verbose: print(i, 'processed.')

        # 创建分子对象
        #gt_m = Chem.MolFromSmiles(gt_m)
        #ot_m = Chem.MolFromSmiles(ot_m)

        # 检查分子是否创建成功
        if gt_m is None or ot_m is None:
            if verbose:
                print(f"Failed to create molecule for descriptor {desc}")
            continue  # 跳过无法创建的分子对

        # 计算指纹和相似性
        MACCS_sim = DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(ot_m), metric=DataStructs.TanimotoSimilarity)
        MACCS_sims.append(MACCS_sim)

        RDK_sim = DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gt_m), Chem.RDKFingerprint(ot_m), metric=DataStructs.TanimotoSimilarity)
        RDK_sims.append(RDK_sim)

        morgan_sim = DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gt_m, morgan_r), AllChem.GetMorganFingerprint(ot_m, morgan_r))
        morgan_sims.append(morgan_sim)

    maccs_sims_score = np.mean(MACCS_sims)
    rdk_sims_score = np.mean(RDK_sims)
    morgan_sims_score = np.mean(morgan_sims)
    if verbose:
        print('Average MACCS Similarity:', maccs_sims_score)
        print('Average RDK Similarity:', rdk_sims_score)
        print('Average Morgan Similarity:', morgan_sims_score)

    return validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score
