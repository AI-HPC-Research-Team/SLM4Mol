import logging
logger = logging.getLogger(__name__)

import warnings

warnings.filterwarnings("ignore", message="A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.")


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.model_manager import MolModel
from utils.xutils import print_model_info, custom_collate_fn, ToDevice
from models import SUPPORTED_Tokenizer, SUPPORTED_CKPT, SUPPORTED_METRIC
from dataset_manager import MolDataset
from accelerate import Accelerator
from torch.optim.lr_scheduler import StepLR

from utils import AverageMeter
import datetime
import time
from transformers import BertTokenizerFast

def task_construct(args):
    input_modal = args.input_modal.split(',')
    if(args.dataset_toy == "toy"):
        task = {
            'name': args.task_name,
            'dataset': args.dataset,
            'train_data_file': f"{args.dataset_folder}train_900.csv",
            'valid_data_file': f"{args.dataset_folder}validation_100.csv",
            'test_data_file': f"{args.dataset_folder}test_100.csv",
            'input_modal': input_modal,
            'text_encoder': args.text_encoder,
            'image_encoder': args.image_encoder,
            'graph_encoder': args.graph_encoder,
            'decoder' : args.decoder,
            'output_modal': [args.output_modal],
            'metric': SUPPORTED_METRIC[args.task_name],
            'image_path':args.image_path
        }
    else:
        task = {
            'name': args.task_name,
            'dataset': args.dataset,
            'train_data_file': f"{args.dataset_folder}train.csv",
            'valid_data_file': f"{args.dataset_folder}validation.csv",
            'test_data_file': f"{args.dataset_folder}test.csv",
            'input_modal': input_modal,
            'text_encoder': args.text_encoder,
            'image_encoder': args.image_encoder,
            'graph_encoder': args.graph_encoder,
            'decoder' : args.decoder,
            'output_modal': [args.output_modal],
            'metric': SUPPORTED_METRIC[args.task_name],
            'image_path':args.image_path
        }
    return task

def encoder_decoder_info(args):
    input_modal = args.input_modal.split(',')   
    text_modals = ['SMILES', 'caption', 'IUPAC', 'SELFIES', 'InChI']
    encoder = ""
    for modal in input_modal:
        if modal in text_modals:
            encoder = f"{encoder}_{args.text_encoder}"
        elif modal == 'graph':
            encoder = f"{encoder}_{args.graph_encoder}"
        elif modal == 'image':
            encoder = f"{encoder}_{args.image_encoder}"
        else:
            continue
        return f"{encoder}_{args.decoder}"
        
        
def encoder_tokenizer(args):
    input_modal = args.input_modal.split(',')
    
    text_encoders = ['SMILES', 'caption', 'IUPAC', 'SELFIES', 'InChI']

    tokenizer_org = None
    if any(encoder in input_modal for encoder in text_encoders):
        tokenizer_org = SUPPORTED_Tokenizer[args.text_encoder].from_pretrained(SUPPORTED_CKPT[args.text_encoder], model_max_length=512)
        tokenizer_org.add_special_tokens({"bos_token": "[DEC]"})
        
    return tokenizer_org
        

def train_mol_decoder(train_loader, valid_loader, test_loader, model, optimizer, scheduler, args, device, task, best_loss = None):
    running_loss = AverageMeter()
    step = 0
    loss_values = {"train_loss": [], "val_loss": [], "test_loss": []}
    last_ckpt_file = None
    if not os.path.exists(f"{args.log_save_path}/{args.task_name}"):
        os.makedirs(f"{args.log_save_path}/{args.task_name}")
    log_file = f"{args.log_save_path}/{args.task_name}/{args.input_modal}{args.encoder_decoder_info}_{args.task_num}_{args.output_modal}"
    with open(f"{log_file}.txt", 'a') as f:
        f.write(str(args) + "\n")
    patience = 0
    for epoch in range(args.epochs):
        logger.info("========Epoch %d========" % (epoch + 1))
        logger.info("Training...")
        #model.train()
        train_loss = []
        train_loader = tqdm(train_loader, desc="Training")
        start_time = time.time()
        for mol in train_loader:
            mol = ToDevice(mol, device)
            loss = model(mol)
            #accelerator.backward(loss)
            loss.backward()
            optimizer.step()
            #scheduler.step()
            optimizer.zero_grad()

            running_loss.update(loss.detach().cpu().item())
            step += 1
            if step % args.logging_steps == 0:
                logger.info("Steps=%d Training Loss=%.4lf" % (step, running_loss.get_average()))
                train_loss.append(running_loss.get_average())
                running_loss.reset()
        end_time = time.time()
        elapsed_time = end_time - start_time
        loss_values["train_loss"].append(np.mean(train_loss))
        val_loss = val_mol_decoder(valid_loader, model, task, device)
        test_loss = val_mol_decoder(test_loader, model, task, device)
        loss_values["val_loss"].append(val_loss)
        loss_values["test_loss"].append(test_loss)

        if not os.path.exists(f"{args.log_save_path}/{args.task_name}"):
                os.makedirs(f"{args.log_save_path}/{args.task_name}")
        if best_loss == None or val_loss<best_loss :
        #if True:
            patience = 0
            best_loss = val_loss
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            if not os.path.exists(f"{args.ckpt_output_path}/{args.task_name}"):
                os.makedirs(f"{args.ckpt_output_path}/{args.task_name}")
            if(epoch>3):
                ckpt_file = f"{args.input_modal}_{args.encoder_decoder_info}_{args.task_num}_{args.output_modal}_{epoch}_{timestamp}.pth"
                ckpt_path = os.path.join(f"{args.ckpt_output_path}/{args.task_name}", ckpt_file)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'best_loss': best_loss
                }, ckpt_path)
                
                message = f"epoch: {epoch}, best_loss:{best_loss} ,val_loss:{val_loss}, time: {elapsed_time:.2f}, {ckpt_file} saved. "
                print(message)
                if last_ckpt_file is not None and os.path.exists(last_ckpt_file):
                    os.remove(last_ckpt_file)
                    print(f"Deleted checkpoint file: {last_ckpt_file}")
                last_ckpt_file = ckpt_path
            else:
                message = f"epoch: {epoch}, best_loss:{best_loss} ,val_loss:{val_loss}, time: {elapsed_time:.2f}, epoch_num < 5, ckpt passed. "
                print(message)
            with open(f"{log_file}.txt", 'a') as f:
                f.write(str(loss_values) + "\n") 
                f.write(message + "\n") 
            print(loss_values)
        else:
            patience = patience+1
            scheduler.step()
            message = f"epoch: {epoch}, best_loss:{best_loss} ,val_loss:{val_loss}, time: {elapsed_time:.2f}, ckpt passed, patience : {patience}. "
            if last_ckpt_file is not None :
                state_dict = torch.load(last_ckpt_file, map_location='cpu')["model_state_dict"]
                best_loss = torch.load(last_ckpt_file, map_location='cpu')["best_loss"]
                model.load_state_dict(state_dict, strict = False)
            metric = test_mol_decoder(test_loader, model, task, device, message)
            message = message + f"epoch {epoch-1} metric : {metric}."
            print(message)
            with open(f"{log_file}.txt", 'a') as f:
                f.write(str(loss_values) + "\n") 
                f.write(message + "\n") 
            print(loss_values)
        if patience > args.patience:
            print("Early stopping due to reaching patience limit.")
            break
            
def val_mol_decoder(valid_loader, model, task, device):
    model.eval()
    val_loss = 0
    logger.info("Validating...")
    with torch.no_grad():
        valid_loader = tqdm(valid_loader, desc="Validation")
        for i, mol in enumerate(valid_loader):
            output_modal = f"{args.output_modal}_truth"
            if output_modal in mol:
                truth = mol[output_modal]
                del mol[output_modal]
            mol = ToDevice(mol, device)
            mol[output_modal] = truth
            loss = model(mol)
            if(i==1):
                #print(f"task : {task}")
                result = model.generate_text(mol)
                for i in range(len(result)):
                    result[i] = result[i].replace("!", "")
                print(f"Truth-{output_modal} : {truth[0]} | Result : {result[0]}")
            val_loss += loss.detach().cpu().item()
        logger.info("validation loss %.4lf" % (val_loss / len(valid_loader)))
    return val_loss / len(valid_loader)

def test_mol_decoder(test_loader, model, task, device, message = None):
    model.eval()
    test_loss = 0
    logger.info("Testing...")
    if not os.path.exists(f"{args.result_save_path}/{args.task_name}"):
        os.makedirs(f"{args.result_save_path}/{args.task_name}")
    result_file = f"{args.result_save_path}/{args.task_name}/{args.input_modal}_{args.encoder_decoder_info}_{args.task_num}_{args.output_modal}.txt"
    tokenizer = BertTokenizerFast.from_pretrained("../../../ckpts/text_ckpts/scibert_scivocab_uncased")
    i = 0
    with torch.no_grad():
        test_loader = tqdm(test_loader, desc="Test")
        logger.info(f"task : {task}")
        org_data_list = []
        truth_list = []
        result_list = []
        for mol in test_loader:
            mol = ToDevice(mol, device)
            if(args.input_modal == 'graph' or args.input_modal =='image'):
                input_modal = 'SMILES'
            else:
                valid_modalities = [modal for modal in task['input_modal'] if modal not in ['graph', 'image']]
                if valid_modalities:
                    input_modal = valid_modalities[0]
                else:
                  
                    input_modal = 'SMILES'  
            org_data = mol[f'{input_modal}_truth']
            org_data_list = org_data_list + org_data
            truth = mol[f'{args.output_modal}_truth']
            truth_list = truth_list + truth
            mol = ToDevice(mol, device)
            result = model.generate_text(mol)
            for i in range(len(result)):
                result[i] = result[i].replace("!", "")
            if(i==1):
                print(f"Truth: {truth[0]} | Result : {result[0]}")
            i=i+1
            result_list = result_list + result
        metric = task["metric"](tokenizer, truth_list, result_list, org_data_list, args)
        with open(result_file, 'a') as f:   
            f.write(str(args) + "\n")
            if(message == None):
                pass
            else:
                f.write(message + "\n") 
            f.write(metric + "\n") 
        print(metric)
        
        return metric
        
    
def add_arguments(parser):
    """

    """
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--task_num", type=str, default="1")
    parser.add_argument("--task_name", type=str, default="molcap")
    parser.add_argument("--graph_encoder", type=str, default="gin")
    parser.add_argument("--text_encoder", type=str, default="")
    parser.add_argument("--image_encoder", type=str, default="swin")
    parser.add_argument("--decoder", type=str, default="molt5")
    parser.add_argument("--input_modal", type=str, default="")
    parser.add_argument("--output_modal", type=str, default="caption")
    parser.add_argument("--prompt", type=str, default=None)   
    parser.add_argument("--fusion_net", type=str, default="add")   
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument('--dataset', type=str, default='ChEBI-20-MM')
    parser.add_argument('--dataset_toy', type=str, default='normal')
    parser.add_argument("--dataset_folder", type=str, default='../../../datasets/ChEBI-20-MM/')
    parser.add_argument("--ckpt_output_path", type=str, default="../../../ckpts/finetune_ckpts")
    parser.add_argument("--model_output_path", type=str, default="../../../output")
    parser.add_argument("--log_save_path", type=str, default="../../../log")
    parser.add_argument("--result_save_path", type=str, default="../../../result")
    parser.add_argument("--latest_checkpoint", type=str, default="../../../ckpts/finetune_ckpts")
    parser.add_argument("--image_path", type=str, default="../../../datasets/ChEBI-20-MM/image")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=300)
    


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    args.encoder_decoder_info = encoder_decoder_info(args)
    print(args)
    #get_models_info(args)
    input_modal = args.input_modal.split(',')               
    if(args.mode == 'model_check'):
        task = task_construct(args)
        model = MolModel(args)
        print_model_info(model)
    if(args.mode == 'data_check'):
        task = task_construct(args)
        tokenizer_org = encoder_tokenizer(args)
        print(tokenizer_org)
        tokenizer_label = SUPPORTED_Tokenizer[args.decoder].from_pretrained(SUPPORTED_CKPT[args.decoder])
        train_dataset = MolDataset(data_path = task['train_data_file'], 
                                  config = None,
                                  split = "train",
                                  tokenizer_org = tokenizer_org,
                                  tokenizer_label = tokenizer_label,
                                  task=task
                                 )
        test_dataset = MolDataset(data_path = task['test_data_file'], 
                                  config = None,
                                  split = "test",
                                  tokenizer_org = tokenizer_org,
                                  tokenizer_label = tokenizer_label,
                                  task=task
                                 )
        print(test_dataset[1])
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers = 8, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers = 8, collate_fn=custom_collate_fn)
        print(f"train_data_test")
        for i, batch in enumerate(train_loader):
            print(f"batch {i}")
        for i, batch in enumerate(test_loader):
            if i >= 2:
                break
            else:
                print(f"batch {i} : {batch}")
                
    if(args.mode == 'encoder_check'):
        task = task_construct(args)
        model = MolModel(args)
        tokenizer_org = encoder_tokenizer(args)
        tokenizer_label = SUPPORTED_Tokenizer[args.decoder].from_pretrained(SUPPORTED_CKPT[args.decoder])
        test_dataset = MolDataset(data_path = task['test_data_file'], 
                                  config = None,
                                  split = "test",
                                  tokenizer_org = tokenizer_org,
                                  tokenizer_label = tokenizer_label,
                                  task=task
                                 )
        #print(test_dataset[1])
        print(f"dataset length {len(test_dataset)}")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 8, collate_fn=custom_collate_fn)
        for i, batch in enumerate(test_loader):
            #print(f"batch {i} : {batch}")   
            embeddings = model.encode_embeddings(batch)
            #print(f"batch_embeddings {i} : {embeddings}")
            for key in embeddings.keys():
                print(f"{key} embeddings size is {embeddings[key].shape}")
            if(len(input_modal)>1):
                fusion_embeddings, _ = model.encode_h(batch)
                print(f"fusion embeddings size is {fusion_embeddings.shape}")  
            print(f"batch_loss {i} : {model(batch)}")
            if i >= 1:
                break
    if(args.mode == 'attention_check'):
        task = task_construct(args)
        model = MolModel(args)
        tokenizer_org = encoder_tokenizer(args)
        tokenizer_label = SUPPORTED_Tokenizer[args.decoder].from_pretrained(SUPPORTED_CKPT[args.decoder])
        test_dataset = MolDataset(data_path = task['test_data_file'], 
                                  config = None,
                                  split = "test",
                                  tokenizer_org = tokenizer_org,
                                  tokenizer_label = tokenizer_label,
                                  task=task
                                 )
        #print(test_dataset[1])
        print(f"dataset length {len(test_dataset)}")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 8, collate_fn=custom_collate_fn)
        for i, batch in enumerate(test_loader):
            #print(f"batch.len {i} : {batch['SMILES']['input_ids'].shape}")   
            embeddings = model.encode_embeddings(batch)
            #print(f"batch_embeddings {i} : {embeddings}")
            for key in embeddings.keys():
                print(f"{key} embeddings size is {embeddings[key].shape}")
            if(len(input_modal)>1):
                fusion_embeddings, _ = model.encode_h(batch)
                print(f"fusion embeddings size is {fusion_embeddings.shape}")  
            attentions = model.get_attentions(batch)
            if attentions is not None:
                for j, attention in enumerate(attentions):
                    print(f"Attention shape for layer {j}: {attention.shape}")
                attention = attentions[0][0]
                print(f"Attention 0 0 : {attention}")
            if i >= 1:
                break
                
    if(args.mode == 'train'):
        task = task_construct(args)
        
        args.latest_checkpoint = None
        best_loss = None
        latest_checkpoint = args.latest_checkpoint
        if args.latest_checkpoint:
            print(f"Latest checkpoint: {latest_checkpoint}")
        else:
            print("No checkpoint found.")
        # model
        logger.info("Loading model ......")
        model = MolModel(args)
        if latest_checkpoint is not None :
            state_dict = torch.load(latest_checkpoint, map_location='cpu')["model_state_dict"]
            best_loss = torch.load(latest_checkpoint, map_location='cpu')["best_loss"]
            model.load_state_dict(state_dict, strict = False)
        print_model_info(model,level=2)
        logger.info("Loading model successed")
        
        # dataset
        logger.info("Loading dataset ......")

        tokenizer_org = encoder_tokenizer(args)
        tokenizer_label = SUPPORTED_Tokenizer[args.decoder].from_pretrained(SUPPORTED_CKPT[args.decoder])
        train_dataset = MolDataset(data_path = task['train_data_file'], 
                                  config = None,
                                  split = "train",
                                  tokenizer_org = tokenizer_org,
                                  tokenizer_label = tokenizer_label,
                                  task=task
                                 )
        valid_dataset = MolDataset(data_path = task['valid_data_file'], 
                                  config = None,
                                  split = "valid",
                                  tokenizer_org = tokenizer_org,
                                  tokenizer_label = tokenizer_label,
                                  task=task
                                 )    
        test_dataset = MolDataset(data_path = task['test_data_file'], 
                                  config = None,
                                  split = "test",
                                  tokenizer_org = tokenizer_org,
                                  tokenizer_label = tokenizer_label,
                                  task=task
                                 )
        #dataloader
        logger.info("Loading dataset successed")

        logger.info("Loading dataloader ......")
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                  pin_memory=True)

        valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                pin_memory=True)
        test_loader = DataLoader(test_dataset, 4, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
        logger.info("Loading dataloader successed")

        #training
       
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
   
        model = model.to(args.device)
        
        print(f"now device is {args.device}")
        
        train_mol_decoder(train_loader, valid_loader, test_loader, model, optimizer, scheduler, args, args.device, task, best_loss)
        
    if(args.mode == 'eval'):
        #task = task_construct(args, "toy")
        task = task_construct(args)
        #args.latest_checkpoint = f"{args.latest_checkpoint}/{args.task_name}/graph_molt5_29_20231020-1332.pth"
        args.latest_checkpoint = None
        best_loss = None
        latest_checkpoint = args.latest_checkpoint
        if args.latest_checkpoint:
            print(f"Latest checkpoint: {latest_checkpoint}")
        else:
            print("No checkpoint found.")
        # model
        logger.info("Loading model ......")
        model = MolModel(args)
        if latest_checkpoint is not None :
            state_dict = torch.load(latest_checkpoint, map_location='cpu')["model_state_dict"]
            best_loss = torch.load(latest_checkpoint, map_location='cpu')["best_loss"]
            model.load_state_dict(state_dict, strict = False)
        print_model_info(model,level=2)
        logger.info("Loading model successed")
        
        # dataset
        logger.info("Loading dataset ......")
        tokenizer_org = encoder_tokenizer(args)
        tokenizer_label = SUPPORTED_Tokenizer[args.decoder].from_pretrained(SUPPORTED_CKPT[args.decoder])
        
        test_dataset = MolDataset(data_path = task['test_data_file'], 
                                  config = None,
                                  split = "test",
                                  tokenizer_org = tokenizer_org,
                                  tokenizer_label = tokenizer_label,
                                  task=task
                                 )

        #dataloader
        logger.info("Loading dataset successed")

        logger.info("Loading dataloader ......")
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                pin_memory=True)
        logger.info("Loading dataloader successed")

        #training
   
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

        model = model.to(args.device)
        print(f"now device is {args.device}")
        test_mol_decoder(test_loader, model, task, args.device)
        