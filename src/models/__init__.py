from models.molecule import *
from models.multimodal import MoMu, SwinTransformer
from models.multimodal.molt5 import MolT5, T5, T511, MT5, MolT5_large
from models.multimodal.biogpt import BioGPT
from models.multimodal.bert import BERT, SciBERT
from models.molecule.roberta import Roberta, ChemBERTa
from models.molecule.bart import BART
from models.molecule.gpt2 import GPT2
from models.molecule.gptneo import GPTNEO
from models.molecule.chemgpt import ChemGPT
from transformers import T5Tokenizer, BertTokenizer, BioGptTokenizer, RobertaTokenizer, BartTokenizer, GPT2Tokenizer, AutoTokenizer
from models.metric import test_caption, test_smiles, test_iupac, test_opt_smi, test_mtr
from models.multimodal.resnet import ResNet
from models.multimodal.vit import ViT
from models.multimodal.swin_nopre import SwinModel

SUPPORTED_MOL_SMILES_ENCODER = {
    "molt5": MolT5,
    "bert": BERT,
    "scibert" : SciBERT,
    "biogpt": BioGPT
    #"git-mol": GIT-Mol
}

SUPPORTED_MOL_GRAPH_ENCODER = {
    #"gin":GIN
    "gin": MoMu,
    "gcn": MoMu,
    "gat": MoMu
    #"git-mol": GIT-Mol
}

SUPPORTED_TEXT_ENCODER = {
    "t5": T5,
    "t511": T511,
    "mt5": MT5,
    "molt5": MolT5,
    "molt5_large": MolT5_large,
    "bart": BART,
    "bert": BERT,
    "scibert" : SciBERT,
    "roberta" : Roberta,
    "chemberta" :ChemBERTa,
    "gpt2": GPT2,
    "gptneo": GPTNEO,
    "chemgpt": ChemGPT, 
    "biogpt": BioGPT
    #"git-mol": GIT-Mol
}

SUPPORTED_IMAGE_ENCODER = {
    "swin": SwinTransformer,
    "swin_nopre": SwinModel,
    "resnet": ResNet,
    "vit": ViT
}

SUPPORTED_DECODER = {
    "t5": T5,
    "t511": T511,
    "mt5": MT5,
    "molt5": MolT5,
    "molt5_large":MolT5_large,
    "bart": BART,
    "gpt2": GPT2,
    "gptneo": GPTNEO,
    "chemgpt": ChemGPT, 
    "biogpt": BioGPT
}

SUPPORTED_Tokenizer = {
    "t5": T5Tokenizer,
    "t511": T5Tokenizer,
    "mt5": T5Tokenizer,
    "molt5": T5Tokenizer,
    "molt5_large" : T5Tokenizer,
    "bart": BartTokenizer,
    "bert": BertTokenizer,
    "scibert" : BertTokenizer,
    "roberta" : RobertaTokenizer,
    "chemberta" : RobertaTokenizer,
    "gpt2": GPT2Tokenizer,
    "gptneo": GPT2Tokenizer,
    "chemgpt": AutoTokenizer,
    "biogpt": BioGptTokenizer
}
ckpt_folder = "/workspace/lpf/CLM-insights/ckpts/"
SUPPORTED_CKPT = {
    "t5": ckpt_folder+"text_ckpts/flan-t5-base",
    "t511": ckpt_folder+"text_ckpts/t5-v1_1-base",
    "mt5": ckpt_folder+"text_ckpts/flan-t5-base",
    "molt5": ckpt_folder+"text_ckpts/molt5-base",
    "molt5_large": ckpt_folder+"text_ckpts/molt5-large" , 
    "bart": ckpt_folder+"text_ckpts/bart",
    "bert": ckpt_folder+"text_ckpts/bert-base-uncased",
    "roberta" : ckpt_folder+"text_ckpts/roberta",
    "chemberta" : ckpt_folder+"text_ckpts/chemberta",
    "swin": ckpt_folder+"image_ckpts/swin_transform_focalloss.pth",
    "swin_nopre":ckpt_folder+"image_ckpts/swin-tiny",
    "momu": ckpt_folder+"fusion_ckpts/momu/MoMuS.ckpt",
    "scibert" : ckpt_folder+"text_ckpts/scibert_scivocab_uncased",
    "gpt2": ckpt_folder+"text_ckpts/gpt2",
    "gptneo": ckpt_folder+"text_ckpts/gptneo",
    "chemgpt": ckpt_folder+"text_ckpts/chemgpt",
    "biogpt": ckpt_folder+"text_ckpts/biogpt",
    "resnet": ckpt_folder+"image_ckpts/resnet-50",
    "vit": ckpt_folder+"image_ckpts/vit-base-patch16-224"
}

SUPPORTED_METRIC = {
    "mol2IUPAC": test_iupac,
    "molcap": test_caption,
    "molopt2smi": test_opt_smi,
    "molopt2IUPAC": test_iupac,
    "molretri":test_mtr,
    "textmolgen": test_smiles,
    "IUPAC2mol": test_smiles,
    "image2smi":test_smiles,
    "image2IUPAC":test_iupac
}
