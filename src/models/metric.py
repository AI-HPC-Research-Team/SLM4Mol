from evaluations.text_translation_metrics import text_evaluate
from evaluations.fingerprint_metrics import molfinger_evaluate
from evaluations.mol_translation_metrics import mol_evaluate, mol_opt_evaluate
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

def test_caption(tokenizer, targets, preds, smiles, args):
    #tokenizer = T5Tokenizer.from_pretrained("../../ckpts/text_ckpts/molt5-large")
    #tokenizer = BertTokenizerFast.from_pretrained("../../ckpts/text_ckpts/scibert_scivocab_uncased")
    bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score, result_dataframe = text_evaluate(tokenizer, targets, preds, smiles, 512)
    message = 'input: {}, Metrics: bleu-2:{}, bleu-4:{}, rouge-1:{}, rouge-2:{}, rouge-l:{}, meteor-score:{}'.format(args.input_modal, bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score)
    #result_dataframe.to_csv(args.result_save_path, index = False)
    if not os.path.exists(f"{args.model_output_path}/{args.task_name}"):
        os.makedirs(f"{args.model_output_path}/{args.task_name}")
    output_save_path = f"{args.model_output_path}/{args.task_name}/{args.input_modal}_{args.encoder_decoder_info}_{args.task_num}_{args.output_modal}.txt"
    with open(output_save_path, 'a') as f:
        result_dataframe.to_csv(f, header=f.tell()==0, sep='\t', index=False)
    return message


def test_smiles(tokenizer, targets, preds, descriptions, args):
    bleu_score, exact_match_score, levenshtein_score, validity_score, result_dataframe = mol_evaluate(targets, preds, descriptions)
    finger_metrics = molfinger_evaluate(targets, preds)
        # print(targets[0], preds[0])
        #fcd_metric= fcd_evaluate(targets, preds)
    message = "input: {}, Metrics: bleu_score:{}, em-score:{}, levenshtein:{}, maccs fts:{}, rdk fts:{}, morgan fts:{}, validity_score:{}".format(args.input_modal, bleu_score, exact_match_score, levenshtein_score, finger_metrics[1], finger_metrics[2], finger_metrics[3], validity_score)
    print(result_dataframe.head())
    if not os.path.exists(f"{args.model_output_path}/{args.task_name}"):
        os.makedirs(f"{args.model_output_path}/{args.task_name}")
    output_save_path = f"{args.model_output_path}/{args.task_name}/{args.input_modal}_{args.encoder_decoder_info}_{args.task_num}_{args.output_modal}.txt"
    with open(output_save_path, 'a') as f:
        result_dataframe.to_csv(f, header=f.tell()==0, sep='\t', index=False)
    return message

def test_iupac(tokenizer, targets, preds, smiles, args):
   
    exact_matches = sum([1 for target, pred in zip(targets, preds) if target == pred]) / len(targets)

    bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score, result_dataframe = text_evaluate(tokenizer, targets, preds, smiles, 512)
    message = 'input: {}, Metrics: bleu-2:{}, bleu-4:{}, rouge-1:{}, rouge-2:{}, rouge-l:{}, meteor-score:{}, exact_matches:{}'.format(args.input_modal, bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score, exact_matches)
    #result_dataframe.to_csv(args.result_save_path, index = False)
    if not os.path.exists(f"{args.model_output_path}/{args.task_name}"):
        os.makedirs(f"{args.model_output_path}/{args.task_name}")
    output_save_path = f"{args.model_output_path}/{args.task_name}/{args.input_modal}_{args.encoder_decoder_info}_{args.task_num}_{args.output_modal}.txt"
    with open(output_save_path, 'a') as f:
        result_dataframe.to_csv(f, header=f.tell()==0, sep='\t', index=False)
    return message

def is_valid_smiles(smiles):
    """Check if a SMILES string is valid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        mol = None
    return mol

def test_mtr(results):
    metrics = {
        'accuracy': 0,
        'MRR': 0,
        'R@1': 0,
        'R@5': 0,
        'R@10': 0
    }
    
    total_queries = len(results)
    
    for query, query_results in results.items():
        truth = query
        
        if len(query_results) > 0 and query_results[0] == truth:
            metrics['accuracy'] += 1
      
        for i, res in enumerate(query_results):
            if res == truth:
                metrics['MRR'] += 1 / (i + 1)
                break
        
        relevant_results = [res for res in query_results if res == truth]
        
        if truth in query_results[:1]:
            metrics['R@1'] += 1
        if truth in query_results[:5]:
            metrics['R@5'] += 1
        if truth in query_results[:10]:
            metrics['R@10'] += 1
    
   
    for metric in metrics:
        metrics[metric] /= total_queries

    return metrics
