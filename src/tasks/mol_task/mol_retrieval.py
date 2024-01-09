import logging
logger = logging.getLogger(__name__)

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
from utils.xutils import print_model_info, custom_collate_fn
from models import SUPPORTED_Tokenizer, SUPPORTED_CKPT, SUPPORTED_METRIC
from dataset_manager import MolDataset
from accelerate import Accelerator
from torch.optim.lr_scheduler import StepLR

from utils import AverageMeter, ToDevice
import datetime
import time
from transformers import BertTokenizerFast
import torch.nn.functional as F
from sklearn.decomposition import PCA

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
        return f"{encoder}_{args.pool}_{args.pool_out}"
    
def encoder_tokenizer(args):

    tokenizer_org = SUPPORTED_Tokenizer[args.text_encoder].from_pretrained(SUPPORTED_CKPT[args.text_encoder], model_max_length=512)
    tokenizer_org.add_special_tokens({"bos_token": "[DEC]"})
        
    return tokenizer_org
        

def train_mol_mtr(train_loader, valid_loader, test_loader, model, optimizer, scheduler, args, device, task, best_loss = None):
    running_loss = AverageMeter()
    step = 0
    loss_values = {"train_loss": [], "val_loss": []}
    last_ckpt_file = None
    if not os.path.exists(f"{args.log_save_path}/{args.task_name}"):
        os.makedirs(f"{args.log_save_path}/{args.task_name}")
    log_file = f"{args.log_save_path}/{args.task_name}/{args.input_modal}{args.encoder_decoder_info}_{args.task_num}_{args.output_modal}"
    with open(f"{log_file}.txt", 'a') as f:
        f.write(str(args) + "\n")
    patience = 0
    message_list = []
    for epoch in range(args.epochs):
        logger.info("========Epoch %d========" % (epoch + 1))
        logger.info("Training...")
        start_time = time.time()
        #model.train()
        train_loss = []
        train_loader = tqdm(train_loader, desc="Training")
        for mol in train_loader:
            mol = ToDevice(mol, device)
            loss = model.forward_mtr(mol)
            loss.backward()
            optimizer.step()
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
        val_loss = val_mol_mtr(valid_loader, model, task, device)
        loss_values["val_loss"].append(val_loss)
        
        if not os.path.exists(f"{args.log_save_path}/{args.task_name}"):
                os.makedirs(f"{args.log_save_path}/{args.task_name}")
            
        if (best_loss == None or val_loss<best_loss) and epoch<25:
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
            message = { 'epoch': epoch, 'best_loss':best_loss ,'val_loss':val_loss, 'time': elapsed_time, 'patience' : patience }
            if last_ckpt_file is not None :
                state_dict = torch.load(last_ckpt_file, map_location='cpu')["model_state_dict"]
                best_loss = torch.load(last_ckpt_file, map_location='cpu')["best_loss"]
                model.load_state_dict(state_dict, strict = False)
            metric = test_mol_retrieve(test_loader, model, args, task, message)
            message ["metric"] = metric
            print(message)
            message_list.append(message)
            with open(f"{log_file}.txt", 'a') as f:
                f.write(str(loss_values) + "\n") 
                f.write(str(message) + "\n") 
            print(loss_values)
        if patience > args.patience:
            print("Early stopping due to reaching patience limit.")
            break
    return message_list
def val_mol_mtr(valid_loader, model, task, device):
    model.eval()
    val_loss = 0
    logger.info("Validating...")
    with torch.no_grad():
        valid_loader = tqdm(valid_loader, desc="Validation")
        for i, mol in enumerate(valid_loader):
            output_modal = 'cid'
            if output_modal in mol:
                truth = mol[output_modal]
                del mol[output_modal]
            mol = ToDevice(mol, device)
            mol[output_modal] = truth
            loss = model.forward_mtr(mol)
            val_loss += loss.detach().cpu().item()
        logger.info("validation loss %.4lf" % (val_loss / len(valid_loader)))
    return val_loss / len(valid_loader)

def retrieve(query, database_ids, database_embeddings, model, top_k=10):
    # Encode the query molecule
    query_embedding = model.encode_mtr_vector(query)['input_modal']
    
    # If query_embedding is 1D, make it 2D (batch_size x embedding_size)
    if query_embedding.dim() == 1:
        query_embedding = query_embedding.unsqueeze(0)
    
    # Move query_embedding to CPU
    query_embedding = query_embedding.cpu()
    
    # Calculate similarities for each database embedding
    similarities = []
    for db_emb in database_embeddings:
        # Ensure db_emb is a tensor and on CPU
        if not isinstance(db_emb, torch.Tensor):
            db_emb = torch.tensor(db_emb)
        db_emb = db_emb.cpu()

        # If db_emb is 1D, make it 2D (1 x embedding_size)
        if db_emb.dim() == 1:
            db_emb = db_emb.unsqueeze(0)
        
        # Calculate cosine similarity and convert to a scalar
        similarity = F.cosine_similarity(query_embedding, db_emb, dim=1)
        # Verify that similarity is a single-value tensor
        if similarity.numel() == 1:
            similarities.append(similarity.item())
        else:
            raise ValueError("Cosine similarity did not return a single value tensor.")

    # Sort the database elements based on similarity
    sorted_indices = sorted(range(len(similarities)), key=lambda k: similarities[k], reverse=True)

    # Return top_k most similar elements
    return [database_ids[i] for i in sorted_indices[:top_k]]

    
def test_mol_retrieve(test_loader, model, args, task, message=None):
    
    model.eval()
    input_modal = args.input_modal
    output_modal = args.output_modal
    #database = [item[f'{output_modal}_labels'] for item in test_dataset]
    database_ids = [item['cid'] for item in test_loader]
    database_embeddings = []
    for mol in tqdm(test_loader, desc="Encoding database"):
        mol = ToDevice(mol, args.device)
        embeddings = model.encode_mtr_vector(mol)['output_modal'] 
        database_embeddings.append(embeddings.detach().cpu().numpy())
    output = {}
    model.eval()
    test_loss = 0
    logger.info("Testing...")
    i = 0
    with torch.no_grad():
        for i, mol in enumerate(tqdm(test_loader, desc="Testing")):
            mol = ToDevice(mol, args.device)
            sim_list = retrieve(mol, database_ids, database_embeddings, model)
            #print(sim_list)
            output[mol['cid'].item()] = sim_list
        metric = task["metric"](output)

        #print(metric)
        return metric
        
def calculate_averages_and_std(message_list):
    
    sum_dict = {
        'epoch': [],
        'best_loss': [],
        'time': [],
        'metric': {}
    }

    
    for message in message_list:
        sum_dict['epoch'].append(message['epoch'])
        sum_dict['best_loss'].append(message['best_loss'])
        sum_dict['time'].append(message['time'])
        for key, value in message['metric'].items():
            if key not in sum_dict['metric']:
                sum_dict['metric'][key] = []
            sum_dict['metric'][key].append(value)

    
    average_std_dict = {}
    for key in ['epoch', 'best_loss', 'time']:
        average_std_dict[key] = {
            'mean': np.mean(sum_dict[key]),
            'std': np.std(sum_dict[key])
        }

    average_std_dict['metric'] = {}
    for key, value_list in sum_dict['metric'].items():
        average_std_dict['metric'][key] = {
            'mean': np.mean(value_list),
            'std': np.std(value_list)
        }

    return average_std_dict
    
def main(args):
    if(args.mode == 'data_check'):
        task = task_construct(args)
        tokenizer_org = encoder_tokenizer(args)
        print(tokenizer_org)
        tokenizer_label = tokenizer_org
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
        print(tokenizer_org)
        tokenizer_label = tokenizer_org
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
        #print(test_dataset[1])
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers = 8, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers = 8, collate_fn=custom_collate_fn)
        for i, batch in enumerate(test_loader):
            #print(f"batch {i} : {batch}")   
            
            embeddings = model.encode_mtr_embeddings(batch)
            #print(f"batch_embeddings {i} : {embeddings}")
            for key in embeddings.keys():
                print(f"batch {i}: {key} embeddings size is {embeddings[key].shape}") 
                #print(f"batch {i}: {key} embeddings size is {embeddings[key]}") 
            
            vectors = model.encode_mtr_vector(batch)
            #print(vectors.keys())
            
            for key in vectors.keys():
                print(f"batch {i}: {key} vectors size is {vectors[key].shape}")
                #print(f"batch {i}: {key} vectors size is {vectors[key]}")
                
            
            print(f"batch_loss {i} : {model.forward_mtr(batch)}")
            
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
        tokenizer_label = tokenizer_org
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
        test_loader = DataLoader(test_dataset, 1, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
        logger.info("Loading dataloader successed")

        #training
        
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        model = model.to(args.device)
        
        print(f"now device is {args.device}")
        
        
        message_list = train_mol_mtr(train_loader, valid_loader, test_loader, model, optimizer, scheduler, args, args.device, task, best_loss)
        return message_list
        
    if(args.mode == 'eval'):

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
        tokenizer_label = None
        
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
        test_loader = DataLoader(test_dataset, 1, shuffle=False, collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                pin_memory=True)
        logger.info("Loading dataloader successed")

        #training
        #accelerator = Accelerator()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        model = model.to(args.device)
        #print(f"now device is {args.device}")
        #model, optimizer, test_loader, scheduler = accelerator.prepare(model, optimizer, test_loader, scheduler)
        
        #test_mol_decoder(test_loader, model, task, args.device)
        result = test_mol_retrieve(test_loader, model, args, task)
        #{cid:['cid1', 'cid2', 'cid3', 'cid4', 'cid5']}
        print(result)

def get_result_info(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read() 
        
        last_model_index = content.rfind('metric')
        if last_model_index == -1:  
            return None
        
        return content[last_model_index:]
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
        
def add_arguments(parser):
    """

    """
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task_num", type=str, default="1")
    parser.add_argument("--task_name", type=str, default="molretri")
    parser.add_argument("--graph_encoder", type=str, default="gin")
    parser.add_argument("--text_encoder", type=str, default="molt5")
    parser.add_argument("--image_encoder", type=str, default="swin")
    parser.add_argument("--decoder", type=str, default="")
    parser.add_argument("--input_modal", type=str, default="SMILES")
    parser.add_argument("--output_modal", type=str, default="caption") 
    parser.add_argument("--prompt", type=str, default=None) 
    parser.add_argument("--pool", type=str, default='avg') 
    parser.add_argument("--pool_out", type=str, default='avg') 
    parser.add_argument('--dataset', type=str, default='ChEBI-20-MM')
    parser.add_argument('--dataset_toy', type=str, default='normal')
    
    parser.add_argument("--dataset_folder", type=str, default='../../../datasets/ChEBI-20-MM/')
    parser.add_argument("--ckpt_output_path", type=str, default="../../../ckpts/finetune_ckpts")
    parser.add_argument("--log_save_path", type=str, default="../../../log")
    parser.add_argument("--result_save_path", type=str, default="../../../result")
    parser.add_argument("--latest_checkpoint", type=str, default="../../../ckpts/finetune_ckpts")
    parser.add_argument("--image_path", type=str, default="../../../datasets/ChEBI-20-MM/image")
    
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=16)
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
    #args.output_modal = None
    encoder_list_0 = ['bert']
    encoder_list_01 = ['bert', 'scibert']
    encoder_list_1 =  ['bert','scibert','roberta','chemberta']
    encoder_list_2 =  ['bart','t5','t511','molt5']
    encoder_list_12 = ['bert','scibert','roberta','chemberta','bart','t5','t511','molt5']
    encoder_list_3 =  ['gin','gcn','gat']
    
    for model_name in encoder_list_12:
        args.text_encoder = model_name
        #for pool in ['avg','max']:
        for pool in ['avg']:
            args.pool = pool
            args.pool_out = pool
  
            args.encoder_decoder_info = encoder_decoder_info(args)
                #message_list = []
            if not os.path.exists(f"{args.result_save_path}/{args.task_name}"):
                os.makedirs(f"{args.result_save_path}/{args.task_name}")
            args.result_file = f"{args.result_save_path}/{args.task_name}/{args.input_modal}_{args.encoder_decoder_info}_{args.output_modal}.txt"
            result = get_result_info(args.result_file)
            if(result is not None):
                print(result)
            else:
                print(args)
                message_list = main(args)
                if(args.mode == 'train'):
                    if(len(message_list)>2):
                        result_dict = calculate_averages_and_std(message_list)
                        print(result_dict)
                        with open(args.result_file, 'a') as f:   
                            f.write(str(args) + "\n")
                            f.write(str(result_dict) + "\n") 
                        print(f"{args.result_file} done!!")
                    else:
                        pass
                else:
                    break
                        
                                        
                        
           

