import logging
logger = logging.getLogger(__name__)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
from tqdm import tqdm
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

import numpy as np
from utils import roc_auc, EarlyStopping, AverageMeter, ToDevice

from models.model_manager import MolModel
from MoleculeNet_loader import MoleculeDataset
from models import SUPPORTED_Tokenizer, SUPPORTED_CKPT
from splitters import scaffold_split, random_split
import pandas as pd
from utils.xutils import print_model_info, custom_collate_fn
from transformers import AutoTokenizer
import time
import gc
from datetime import datetime

def task_construct(args):
    input_modal = args.input_modal.split(',')
    input_modal_filtered = [item for item in input_modal if item != 'graph']
    if(len(input_modal_filtered)>0):
        text_modal = input_modal_filtered[0]
    else:
        text_modal = None
    task = {
        'name': args.task_name,
        'dataset_name': args.dataset_name,
        'input_modal': input_modal,
        'text_modal' :  text_modal,
        'text_encoder': args.text_encoder,
        'graph_encoder': args.graph_encoder,
        'pool': args.pool,
        'dropout': args.dropout,
        'seed':args.seed
    }
    return task

def encoder_tokenizer(args):
    input_modal = args.input_modal.split(',')
    
    text_encoders = ['SMILES', 'caption', 'IUPAC', 'SELFIES', 'InChI']

    tokenizer_org = None
    if any(encoder in input_modal for encoder in text_encoders):
        tokenizer_org = SUPPORTED_Tokenizer[args.text_encoder].from_pretrained(SUPPORTED_CKPT[args.text_encoder], model_max_length=512)
        tokenizer_org.add_special_tokens({"bos_token": "[DEC]"})
    return tokenizer_org
        

def add_arguments(parser):
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task_name", type=str, default="mpp")
    parser.add_argument("--mode", type=str, default="train")
    #parser.add_argument('--dataset', type=str, default='MoleculeNet')
    parser.add_argument("--dataset_folder", type=str, default='../../../datasets/mpp')
    parser.add_argument("--dataset_name", type=str, default='bbbp')
    parser.add_argument("--split", type=str, default='scaffold')
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--param_key", type=str, default="None")
    parser.add_argument("--output_path", type=str,
                        default="../../../ckpts/finetune_ckpts/mpp")
    parser.add_argument("--input_modal", type=str, default="SMILES")
    parser.add_argument("--output_modal", type=str, default=None)
    parser.add_argument("--text_encoder", type=str, default="molt5")
    parser.add_argument("--graph_encoder", type=str, default=None)
    parser.add_argument("--decoder", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None) 
    parser.add_argument("--pool", type=str, default='avg') 
    parser.add_argument("--dropout", type=float, default=0.5) 
    parser.add_argument("--log_save_path", type=str, default="../../../log/mpp")
    parser.add_argument("--result_save_path", type=str, default="../../../result/mpp")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser

def get_metric(args):
    if args.task_num > 0:
        metric_name = "roc_auc"
        metric = roc_auc
    elif args.task_num == 0:
        metric_name = "MSE"
        metric = mean_squared_error
    return metric_name, metric

def get_num_task(dataset):
    dataset = dataset.lower()
    """ used in molecule_finetune.py """
    if dataset == 'tox21':
        return 12
    elif dataset in ['hiv', 'bace', 'bbbp']:
        return 1
    elif dataset == 'muv':
        return 17
    elif dataset == 'toxcast':
        return 617
    elif dataset == 'sider':
        return 27
    elif dataset == 'clintox':
        return 2
    elif dataset in ['esol', 'lipophilicity', 'freesolv']:
        return 0
    raise ValueError('Invalid dataset name.')

dataset_list = ['bace', 'bbbp', 'tox21' , 'toxcast', 'sider', 'clintox','esol', 'lipophilicity', 'freesolv']

def train_mpp(train_loader, valid_loader, test_loader, model, args):
    device = torch.device(args.device)
    if args.task_num > 0:
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        mode = "higher"
    else:
        loss_fn = nn.MSELoss()
        mode = "lower"

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if not os.path.exists(f"{args.output_path}/{args.dataset_name}"):
        os.makedirs(f"{args.output_path}/{args.dataset_name}")
    stopper = EarlyStopping(
        mode=mode, patience=args.patience, filename=f"{args.output_path}/{args.dataset_name}/{args.dataset_name}.pth")
    metric_name = args.metric_name
    metric = args.metric
    running_loss = AverageMeter()
    log_folder = f"{args.log_save_path}/{args.dataset_name}"
    if(os.path.exists(log_folder)):
        pass
    else:
        os.makedirs(log_folder)
    with open(f"{log_folder}/{args.input_modal}_{args.text_encoder}.txt", 'a') as f:
        f.write(str(args) + "\n") 
        
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    test_result_dict_list = []
    start_time = time.time()
    for epoch in range(args.epochs):
        logger.info("========Epoch %d========" % (epoch + 1))
        model.train()
        running_loss.reset()
        epoch_iter = tqdm(train_loader, desc="Iteration")
        training = True
        for step, batch in enumerate(epoch_iter):
            #print(batch)
            batch = ToDevice(batch, device)
            pred = model.forward_mpp(batch)
            #print(pred.shape)
            #print(batch['graph'].y.shape)
            y = batch['label'].view(pred.shape).to(torch.float64).to(device)

            # Loss matrix
            if args.task_num > 0:
                is_valid = y**2 > 0
                loss_mat = loss_fn(pred.double(), (y+1)/2)
                loss_mat = torch.where(is_valid, loss_mat, torch.zeros(
                loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            else:
                is_valid = ~torch.isnan(y)
                loss_mat = loss_fn(pred.double(), y.double())
                
            # loss matrix after removing null target
            optimizer.zero_grad()
            loss = torch.sum(loss_mat)/torch.sum(is_valid)
            loss.backward()
            
            del loss_mat
            optimizer.step()
            epoch_iter.set_description(f"Epoch: {epoch} tloss: {loss:.4f}")
            
            
            running_loss.update(loss.detach().cpu().item())
            if (step + 1) % args.logging_steps == 0:
                logger.info("Steps=%d Training Loss=%.4lf" %
                            (step, running_loss.get_average()))
                running_loss.reset()

        val_metrics_dict = val_mpp(valid_loader, model, args)
        if args.task_num > 0:
            val_metrics = val_metrics_dict['roc_auc']
        else:
            val_metrics = val_metrics_dict['mse']
        
        test_metrics_dict = val_mpp(test_loader, model, args)
        if args.task_num > 0:
            test_metrics = test_metrics_dict['roc_auc']
        else:
            test_metrics = test_metrics_dict['mse']
            
        if args.adapt and args.batch_size<16:
            args.batch_size = int(args.batch_size*2)
            args.logging_steps = int(args.logging_steps/2)
            print(f"Increase batch_size to {args.batch_size}")
            break
            
        val_acc_list.append(val_metrics)
        test_acc_list.append(test_metrics)
        test_result_dict_list.append(test_metrics_dict)
        
        val_log_message = f"epoch {epoch} ,{args.dataset_name}, {metric_name}, {val_metrics}"
        test_log_message = f"epoch {epoch} ,{args.dataset_name}, {metric_name}, {test_metrics}"
        logger.info(val_log_message)
        logger.info(test_log_message)
        with open(f"{log_folder}/{args.input_modal}_{args.text_encoder}.txt", 'a') as f:
            f.write(val_log_message + "\n") 
            f.write(test_log_message + "\n") 
            
        early_stop, avg_score, std_dev ,ckpt_name = stopper.step((val_metrics), model)
        if early_stop :
            args.continue_training = False
            if(args.task_num > 0):

                end_time = time.time()
                elapsed_time = end_time - start_time
                result_epochs.append(val_acc_list.index(max(val_acc_list)))
                result_performance.append(test_acc_list[val_acc_list.index(max(val_acc_list))])
                result_dict.append(test_result_dict_list[val_acc_list.index(max(val_acc_list))])
                result_times.append(elapsed_time)
                with open(args.result_file, "a") as result_file:
                    result_file.write(str(args)+ "\n")
                    result_file.write(f"epoch: {val_acc_list.index(max(val_acc_list))}, val: {max(val_acc_list)*100:.2f}%, test: {test_acc_list[val_acc_list.index(max(val_acc_list))]*100:.2f}%, result_dict: {test_result_dict_list[val_acc_list.index(min(val_acc_list))]}, time: {elapsed_time:.2f}, Average score: {avg_score*100:.2f}% ~ {std_dev*100:.2f}% \n") 
                break
                
            else:
                end_time = time.time()
                elapsed_time = end_time - start_time
                result_epochs.append(val_acc_list.index(min(val_acc_list)))
                result_performance.append(test_acc_list[val_acc_list.index(min(val_acc_list))])
                result_dict.append(test_result_dict_list[val_acc_list.index(min(val_acc_list))])
                result_times.append(elapsed_time)
                with open(args.result_file, "a") as result_file:
                    result_file.write(str(args)+ "\n")
                    result_file.write(f"epoch: {val_acc_list.index(min(val_acc_list))}, val: {min(val_acc_list)*100:.2f}%, test: {test_acc_list[val_acc_list.index(min(val_acc_list))]*100:.2f}%, result_dict: {test_result_dict_list[val_acc_list.index(min(val_acc_list))]}, time: {elapsed_time:.2f}, Average score: {avg_score*100:.2f}% ~ {std_dev*100:.2f}% \n") 
                    
                break
                
    return model, epoch


def val_mpp(valid_loader, model, args):
    device = torch.device(args.device)
    model.eval()

    all_preds, y_true ,id_list= [], [], []

    epoch_iter = tqdm(valid_loader, desc="Iteration")
    
    # Ensure no computation graph is retained
    with torch.no_grad():
        for step, batch in enumerate(epoch_iter):
            batch = ToDevice(batch, device)
            pred = model.forward_mpp(batch)
            label = batch['label'].view(pred.shape).to(device)  # Removed .to(torch.float64)

            all_preds.append(pred.detach())
            y_true.append(label.detach())
            id_list.append(batch['id'].detach())
            
        # Stack tensors directly on the GPU
        all_preds = torch.cat(all_preds, dim=0)
        y_true = torch.cat(y_true, dim=0)
        id_list =  torch.cat(id_list, dim=0)
        
        # Move only the final results to CPU
        all_preds = all_preds.cpu().numpy()
        y_true = y_true.cpu().numpy()
        
        N = len(y_true[0]) if len(y_true) > 0 else 0
        id_list = id_list.cpu().numpy()
        
        extended_id_list = np.repeat(id_list, N).reshape(len(id_list), N)
        
    all_preds_output = all_preds.flatten() 
    y_true_output = y_true.flatten()
    id_list_output = extended_id_list.flatten()
    
    
    """
    print(len(y_true))
    print(len(all_preds))
    print(len(id_list_output))
    """
    #print(y_true_output)
    #print(all_preds_output)
    #print(id_list_output)
    
    len(id_list_output) == len(all_preds_output) == len(y_true_output)

    try:
        output_df = pd.DataFrame({
            'id': id_list_output,
            'Predictions': all_preds_output,
            'True Values': y_true_output
        })
    except ValueError as e:
        print(f"Error: {e}")
        print("Check if 'id_list_output', 'all_preds_output', and 'y_true_output' are one-dimensional and have the same length.")
   
    output_df.to_csv(args.output_file, index=False, mode='w')
    
    roc_list = []
    mse_list = []
    metric_list = []
    #print("xxx")
    #print(args.task_num)
    if(args.task_num > 0):
        roc_list = []
        prc_list = []
        f1_list = []
        mcc_list = []
        ap_list = []
        average_type = 'macro' 
        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive and one negative data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
                is_valid = y_true[:, i]**2 > 0  # Only consider non-zero targets
                # Adjust labels to be in {0, 1}
                labels = (y_true[is_valid, i] + 1) / 2
                preds = all_preds[is_valid, i]
                # Calculate AUC-ROC
                roc_list.append(roc_auc_score(labels, preds))
    
                # Calculate AUC-PR
                prc_list.append(average_precision_score(labels, preds))
    
                # Binarize predictions for F1 and MCC calculation
                preds_min = preds.min()
                preds_max = preds.max()
                if(preds_max-preds_min<0.1):
                    normalized_preds = (preds - preds_min) / (preds_max - preds_min)

                    binarized_preds = (normalized_preds > 0.5).astype(int)
                else:
                    binarized_preds = (preds > 0.5).astype(int)
                
                # Calculate F1
                f1_list.append(f1_score(labels, binarized_preds, average=average_type))
    
                # Calculate MCC
                mcc_list.append(matthews_corrcoef(labels, binarized_preds))
    
                # Calculate AP
                ap_list.append(average_precision_score(labels, all_preds[is_valid, i]))

        if len(roc_list) < y_true.shape[1]:
            print("Some target is missing!")
            print("Missing ratio: %f" % (1 - float(len(roc_list))/y_true.shape[1]))
            
        metric_dict = {
            'roc_auc': sum(roc_list) / len(roc_list),
            'pr_auc': sum(prc_list) / len(prc_list),
            'f1_score': sum(f1_list) / len(f1_list),
            'mcc':sum(mcc_list) / len(mcc_list),
            'ap': sum(ap_list) / len(ap_list)
        }
    else:
        mse_list = []
        rmse_list = []
        mae_list = []
        r2_list = []
        explained_variance_list = []
        
        for i in range(y_true.shape[1]):
            is_valid = ~np.isnan(y_true[:, i])  # Ensure there are no NaNs
            mse = mean_squared_error(y_true[is_valid, i], all_preds[is_valid, i])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true[is_valid, i], all_preds[is_valid, i])
            r2 = r2_score(y_true[is_valid, i], all_preds[is_valid, i])
            explained_variance = explained_variance_score(y_true[is_valid, i], all_preds[is_valid, i])
        
            mse_list.append(mse)
            rmse_list.append(rmse)
            mae_list.append(mae)
            r2_list.append(r2)
            explained_variance_list.append(explained_variance)

        if len(mse_list) < y_true.shape[1]:
            print("Some target is missing!")
            print("Missing ratio: %f" % (1 - float(len(mse_list))/y_true.shape[1]))
        metric_dict = {
            'mse': sum(mse_list)/len(mse_list),
            'rmse': sum(rmse_list)/len(rmse_list),
            'mae': sum(mae_list)/len(mae_list),
            'r2': sum(r2_list)/len(r2_list),
            'explained_variance': sum(explained_variance_list)/len(explained_variance_list)
        }

    return metric_dict  # y_true.shape[1]

    # return {metric_name: metric(all_y, all_preds)}
    
def dataset_construct(args, task):
    tokenizer_org = encoder_tokenizer(args)
    if(tokenizer_org is not None):
        print(f"tokenizer_org is")
        print(tokenizer_org)
    else:
        print("No tokenizer_org")
    mpp_dataset = MoleculeDataset(f"{args.dataset_folder}/{args.dataset_name}", dataset=args.dataset_name)
    smiles_list = pd.read_csv(f"{args.dataset_folder}/{args.dataset_name}/processed/{args.dataset_name}.csv")["SMILES"].tolist()
    mol_dataframe = pd.read_csv(f"{args.dataset_folder}/{args.dataset_name}/processed/{args.dataset_name}.csv")
    if(task['text_modal'] == None):
        text_list = None
    else:
        text_list = mol_dataframe[task['text_modal']]
    if args.split == "scaffold":
        train_dataset, valid_dataset, test_dataset = scaffold_split(mpp_dataset, tokenizer_org, smiles_list, task['text_modal'], text_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(mpp_dataset, tokenizer_org, smiles_list, task['text_modal'], text_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
    else:
        raise ValueError("Invalid split option.")
    print(f"data split {args.split}")
    
    input_modal = args.input_modal.split(',')
    if('graph' not in input_modal):
        for dataset in [train_dataset, valid_dataset, test_dataset]:
            for entry in dataset:
                if 'graph' in entry:
                    del entry['graph']
    return train_dataset, valid_dataset, test_dataset

def main(args):
    # prepare dataset
    input_modal = args.input_modal.split(',')
    task = task_construct(args)
    #set up dataset
    print(task)

    # configure metric
    args.task_num = get_num_task(args.dataset_name)
    
    args.metric_name, args.metric = get_metric(args)

    # TODO: support two and multiple classification
    if(args.mode == 'data_check'):
        train_dataset, valid_dataset, test_dataset = dataset_construct(args, task)
        print(test_dataset[3])
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
        for i, batch in enumerate(train_loader):
            print(f"batch {i}")
        for i, batch in enumerate(test_loader):
            if i >= 2:
                break
            else:
                print(f"batch {i} : {batch}")
        
    elif(args.mode == "encoder_check"):
        train_dataset, valid_dataset, test_dataset = dataset_construct(args, task)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
        model = MolModel(args)
        print_model_info(model,level=2)
        for i, batch in enumerate(train_loader):
            h, _ = model.encode_h(batch)
            embeddings = model.get_mpp_embeddings(h)
            #print(f"batch_embeddings {i} : {embeddings}")
            print(f"embeddings {i} : {embeddings}")
            print(f"embeddings size is {embeddings.shape}")
            print(f"result {i} : {model.forward_mpp(batch)}")
            print(f"result.shape {i} : {model.forward_mpp(batch).shape}")
            if i >= 1:
                break
        
    elif(args.mode == "train"):  
        train_dataset, valid_dataset, test_dataset = dataset_construct(args, task)

        print("Loading model ......")
        device = torch.device(args.device)
        model = MolModel(args)
        print_model_info(model, 2)
        model = model.to(device)
        while args.continue_training:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn) 
            model, epoch = train_mpp(train_loader, valid_loader,
                                    test_loader, model, args)

    elif args.mode == "eval":
        train_dataset, valid_dataset, test_dataset = dataset_construct(args, task)
        #train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
        model = MolModel(args)
        print_model_info(model, 2)
        device = torch.device(args.device)
        model = model.to(device)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
        results = val_mpp(test_loader, model, args)
        logger.info("%s test %s=%.4lf" % (args.dataset_name, args.metric_name, results))
        
def get_result_info(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()  
      
        last_model_index = content.rfind('model')
        if last_model_index == -1:  
            return None
      
        return content[last_model_index:]
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
        
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    # set seed
    #random.seed(args.seed)
        
    if args.split == "scaffold":
        args.log_save_path = "../../../log/mpp"
        args.result_save_path ="../../../result/mpp"
        args.output_save_path = "../../../output/mpp"
    elif args.split == "random":
        args.log_save_path = "../../../log/mpp_random_split"
        args.result_save_path ="../../../result/mpp_random_split"
        
    encoder_list_0 = ['t511']
    encoder_list_1 =  ['bert','scibert','roberta','chemberta']
    encoder_list_2 =  ['bart','t5','t511','molt5']
    encoder_list_12 = ['bert','scibert','roberta','chemberta','bart','t5','t511','molt5']
    encoder_list_3 =  ['gin','gcn','gat']
    result_folder = f"{args.result_save_path}/{args.dataset_name}"
    output_folder = f"{args.output_save_path}/{args.dataset_name}"
    if(os.path.exists(result_folder)):
        pass
    else:
        os.makedirs(result_folder)
        
    if(os.path.exists(output_folder)):
        pass
    else:
        os.makedirs(output_folder)
        
    for model_name in encoder_list_12:
        args.text_encoder = model_name
        for pool_name in ['avg']:
        #for pool_name in ['avg','max']:
            args.pool = pool_name
            for num in [6]: # MLP layers
                args.mlp_layers_num = num
                result_epochs = []
                result_performance = []
                result_dict = []
                result_times = []
                
                args.result_file = f"{result_folder}/{args.input_modal}_{args.text_encoder}_{args.graph_encoder}_{args.pool}_{args.mlp_layers_num}.txt"
                args.output_file = f"{output_folder}/{args.input_modal}_{args.text_encoder}_{args.graph_encoder}_{args.pool}_{args.mlp_layers_num}.txt"
                result = get_result_info(args.result_file)
                
                if(result is not None):
                    print(result)
                else:
                    args.batch_size = 8
                    args.logging_steps = 400
                    for seed in [42]:
                        args.continue_training = True  # criteria to stop the training
                        args.adapt = True  # while this is true, the algorithm will perform batch adaptation
                        args.seed = seed
                        np.random.seed(args.seed)
                        torch.manual_seed(args.seed)
                        torch.cuda.manual_seed(args.seed)
                        main(args)
                    if(len(result_performance)>0):
                        summary_dict = {}
                        keys = result_dict[0].keys()
                        for key in keys:
                            values = [d[key] for d in result_dict]  
                            summary_dict[key + '_mean'] = np.mean(values) 
                            summary_dict[key + '_std'] = np.std(values)  
                        message = f"model: {args.text_encoder}_{args.graph_encoder}, input_modal: {args.input_modal}, pool:{args.pool}, mlp_layers_num:{args.mlp_layers_num}, dropout: {args.dropout}, batch_size: {args.batch_size}, best_val_epoch : {np.mean(result_epochs)}~{np.std(result_epochs)}, test_perfomance : {np.mean(result_performance)}~{np.std(result_performance)}, result: {summary_dict} , time: {np.mean(result_times)}"
                        print(message)
                        with open(args.result_file, "a") as result_file:
                            result_file.write(str(message)+ "\n")
                print(f"{args.result_file} finish！")
                
    print(f"{result_folder}/{args.input_modal}_{args.text_encoder}_{args.graph_encoder} all task finish")
    