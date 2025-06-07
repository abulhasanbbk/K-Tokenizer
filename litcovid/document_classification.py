#!/usr/bin/env python
# coding=utf-8
"""
Author: Abul Hasan
LitCOVID docuemnt classification task using K-tokenizer and variable sizes of training.
There are several models we will try here:
(i) PubmedBERT
(ii) BioFormer
The BioFormer has their own vocabulary and model weights: This link is as following:
bioformers/bioformer-16L
When we use K-Tokenizer that will produce another two models
 (i) Pubmed_k_bert
 (ii) BioFormer_k_bert
And we will try using 20%, 30%, and 50% training dataset.

"""
import pandas as pd
import os
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import numpy as np
from transformers import AutoModelForSequenceClassification, set_seed, TrainingArguments, Trainer, EvalPrediction, TrainerCallback
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import torch
import pickle
import argparse
from utils import *
CURRENT_EPOCH = 0

def add_doc_id_column(hf_ds, file_tag):
    """
    Adds a deterministic doc_id column '<fileTag>::rowIdx' so that
    predictions from baseline and K-Tokeniser can be aligned later.
    """
    return hf_ds.add_column(
        "doc_id",
        [f"{file_tag}::{i}" for i in range(len(hf_ds))]
    )

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Folder created:", folder_path)
    else:
        print("Folder already exists:", folder_path)


# ----------------- metrics ---------------------------------------------------
def multi_label_metrics(pred_logits, gold_labels, id2label):
    sigmoid  = torch.nn.Sigmoid()
    probs    = sigmoid(torch.tensor(pred_logits)).numpy()
    y_pred   = (probs >= 0.5).astype(int)
    y_true   = gold_labels

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(  y_true, y_pred, average="macro", zero_division=0)
    macro_f1= f1_score(      y_true, y_pred, average="macro", zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    acc     = accuracy_score(y_true, y_pred)

    return {
        "f1_micro":   micro_f1,              # <- used for early stopping
        "p_micro":    micro_p,
        "r_micro":    micro_r,
        "f1_macro":   macro_f1,
        "p_macro":    macro_p,
        "r_macro":    macro_r,
        "roc_auc":    roc_auc,
        "accuracy":   acc,
    }

def hf_compute_metrics(pred: EvalPrediction, id2label):
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    return multi_label_metrics(logits, pred.label_ids, id2label)

"""Process the dataset to get labels in 0 or 1 format
"""
def preprocess_data(examples, tokenizer, labels_list, label2id, k_tok):
    """
    examples : dict with keys like
        "abstract", "label", "doc_id", ...
    """
    text   = examples["abstract"]
    doc_id = examples["doc_id"]

    # ---------------- tokenisation ------------------
    if k_tok in ("umls", "pubmed"):
        enc = tokenizer(text, batch=True)
    else:
        enc = tokenizer(text, padding="max_length",
                        truncation=True, max_length=512)

    # ---------------- labels (train / val batches only) ----------
    if "label" in examples:                     # <-- fixed line
        L = np.zeros((len(text), len(labels_list)))
        for i, lbls in enumerate(examples["label"]):
            for lb in lbls.split(";"):
                L[i, label2id[lb]] = 1
        enc["labels"] = L.tolist()

    enc["doc_id"] = doc_id                     # keep IDs
    return enc


def create_hugging_face_dataset(filename):
    df = pd.read_csv(filename)
    missing_rows = df.isna().any(axis=1)
    #print(missing_rows)
    df = df.dropna()
    #df=df[0:10]
    hf_dataset=Dataset.from_pandas(df)
    return hf_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parse arguments
    parser.add_argument("--type", type=str, required=True,
                        help="p for pubmed b for bioformer ")
    parser.add_argument("--tokenizer_base_loc", type=str, required=True,
                        help="Provide locations where all your tokenizer file resides ")
    parser.add_argument("--corpus", type=str, required=True,
                        help="Provide corpus type:umls or pubmed")
    parser.add_argument("--alpha", type=int, required=True,
                        help="Provide alpha parameter")
    parser.add_argument("--delta", type=int, required=True,
                        help="Provide delta parameter")
    parser.add_argument("--partition", type=int, required=True,
                        help="Provide partition of the training dataset, 1 for 100%")
    parser.add_argument("--seed", type=int, required=True,
                        help="Provide 42, 2046, or 1234")
    parser.add_argument("--model_save_loc", type=str, required=True,
                        help="Provide saving location for model")
    parser.add_argument("--numrun", type=int, required=True,
                        help="Provide run number, 1,2,or 3")
    args = parser.parse_args()
    set_seed(args.seed)
    alpha=args.alpha
    delta=args.delta
   
    if args.corpus=="umls" and args.type=="p":
        fileNames=PubmedBERTFromUMLSFileNames()
        infusedBertModelFileName=InfusedPubmedFromUMLSFileNames()
        result_base_loc="pubmed_bert_umls" 
        model_dir_loc= "model_pubmed_bert_umls" 
    elif args.corpus=="pubmed" and args.type=="p":
        fileNames=PubmedBERTFromPubMedFileNames()
        infusedBertModelFileName=InfusedPubmedFromPubMedFileNames()
        result_base_loc="pubmed_bert_pubmed" 
        model_dir_loc= "model_pubmed_bert_pubmed" 
    elif args.corpus=="pubmed" and args.type=="b":
        fileNames=BioformerFromPubMedFileNames()
        infusedBertModelFileName=InfusedBioformerFromPubMedFileNames()
        result_base_loc="bioformer_bert_pubmed"  
        model_dir_loc= "model_bioformer_bert_pubmed" 
    elif args.corpus=="umls" and args.type=="b":
        fileNames=BioformerFromUMLSFileNames()
        infusedBertModelFileName=InfusedBioformerFromUMLSFileNames()
        result_base_loc="bioformer_bert_umls" 
        model_dir_loc= "model_bioformer_bert_umls" 
    elif args.corpus=="base" and args.type=="b":
        print("bioformer-base")
        result_base_loc="bioformer_bert" 
        model_dir_loc="model_bioformer_bert" 
    elif args.corpus=="base" and args.type=="p":
        print("pubmed-bert")
        result_base_loc="pubmed_bert" 
        model_dir_loc="model_pubmed_bert"
    else:
        pass
    
    if args.corpus=="umls" or args.corpus=="pubmed":
        locations=Loc()
        locations.base_dir=args.tokenizer_base_loc
        locations.base=fileNames.base_tok
        locations.disease=os.path.join(args.tokenizer_base_loc, fileNames.base_loc_disease+str(alpha)+"-"+str(delta))
        locations.drug=os.path.join(args.tokenizer_base_loc, fileNames.base_loc_drug+str(alpha)+"-"+str(delta))
        locations.ktok=os.path.join(args.tokenizer_base_loc, fileNames.base_loc_ktok+str(alpha)+"-"+str(delta))
    else:
        pass
    
    #bioformer_model_loc="bioformers/bioformer-16L"
    batch_size = 4
    metric_name = "f1"
    #out_dir="pubmed_k_pubmed_models" #model output directory
    r_loc="Results-3-Runs-50-percent"
    create_folder_if_not_exists(r_loc)
    create_folder_if_not_exists(args.model_save_loc)
    #create_folder_if_not_exists(out_dir)
    #create_folder_if_not_exists(result_base_loc)
    result_loc1=os.path.join(r_loc,result_base_loc)
    create_folder_if_not_exists(result_loc1)
    #create_folder_if_not_exists(model_loc1)
    run_loc="run_"+str(args.numrun)
    result_loc=os.path.join(result_loc1, run_loc)
    create_folder_if_not_exists(result_loc)
    #create_folder_if_not_exists(model_loc)
    train_file="data/BC7-LitCovid-Train.csv"
    validation_file="data/BC7-LitCovid-Dev.csv"
    classification_type="multi_label_classification"
    val10_file = "data/BC7-LitCovid-Dev_10p.csv"  #test files do not have labels. So I am taking validation file as test dataset
    #tokenizer = AutoTokenizer.from_pretrained(bioformer_model_loc)
    
   
    label2id={'Epidemic Forecasting':0,
          'Treatment':1,
          'Prevention':2, 
          'Mechanism':3,
          'Case Report':4, 
          'Transmission':5,
          'Diagnosis':6
         }
    id2label = {label2id[label]:label for label in label2id}
    labels_list=[label2id[idx] for idx in label2id]
    print(labels_list)
    train_ds = add_doc_id_column(create_hugging_face_dataset(train_file), "train")
    val_ds   = add_doc_id_column(create_hugging_face_dataset(val10_file), "val10")
    test_ds  = add_doc_id_column(create_hugging_face_dataset(validation_file), "test")
    num_samples = len(train_ds) #total number of samples
    print(num_samples)
    num_parts =args.partition # number of parts for 20% (5; discarding extra parts after integer division ), 30% (3), 50%
    part_size = num_samples // num_parts
    parts = []
    for i in range(num_parts):
        start_idx = i * part_size
        end_idx = (i + 1) * part_size
        part = train_ds.select(list(range(start_idx, end_idx)))
        parts.append(part)
    
    #test_dataset=test_dataset[0:10]
    if args.partition==1:
        partition_str="100_percent"
    elif args.partition==2:
        partition_str="50_percent"
    elif args.partition==3:
        partition_str="30_percent"
    elif args.partition==5:
        partition_str="20_percent"
    else:
        pass
    fertilities=[0.0]
    for f in fertilities: 
        for idx, ds in enumerate(parts):
            if args.corpus=="umls":
                bert_model_loc=infusedBertModelFileName.infused_bert_model+str(alpha)+"-"+str(delta)
                tokenizer=KTokenizerFromUMLS(locations, f)
            elif args.corpus=="pubmed":
                bert_model_loc=infusedBertModelFileName.infused_bert_model+str(alpha)+"-"+str(delta)
                tokenizer=KTokenizerFromPubMed(locations, f)
            elif args.corpus=="base": 
                if args.type=="p":
                    bert_model_loc="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
                    tokenizer=AutoTokenizer.from_pretrained(bert_model_loc)
                    print("Auto Tokenizer Loaded from:", bert_model_loc)
                    #ins=input("Continue?")
                elif args.type=="b":
                    bert_model_loc="bioformers/bioformer-16L"
                    tokenizer=AutoTokenizer.from_pretrained(bert_model_loc)
                    print("Auto Tokenizer Loaded from", bert_model_loc)
                    #ins=input("Continue?")

            
            #print (ds[0])
            #create_folder_if_not_exists(out_dir) #create model output location
            if args.corpus=="umls" or args.corpus=="pubmed":
                f_result_dir=partition_str+"_f_"+str(f*1000)
                f_model_dir=partition_str+"_f_"+str(f*1000)
            elif args.corpus=="base":
                f_result_dir=partition_str+"_f_000"
                f_model_dir=partition_str+"_f_000"
            create_folder_if_not_exists(os.path.join(result_loc, f_result_dir))
            part_dir="part_"+str(idx)
            result_part_dir_loc=os.path.join(os.path.join(result_loc, f_result_dir), part_dir)
            create_folder_if_not_exists(result_part_dir_loc)
            model_save=os.path.join(args.model_save_loc, "Run_"+ str(args.numrun))
            create_folder_if_not_exists(model_save)
            model_part_dir_loc=os.path.join(model_save, part_dir)
            create_folder_if_not_exists(model_part_dir_loc)
            #create training encoding with the part training data
            train_encoding=train_ds.map(preprocess_data, 
                                  fn_kwargs={'tokenizer': tokenizer, 'labels_list':labels_list, 'label2id':label2id, 'k_tok':args.corpus},
                                  batched=True,
                                  remove_columns=train_ds.column_names
                                  )
        
            #initilize the model

            test_encoding=test_ds.map(preprocess_data,  
                                           fn_kwargs={'tokenizer': tokenizer,'labels_list':labels_list,'label2id':label2id,'k_tok':args.corpus},
                                           batched=True,
                                           remove_columns=test_ds.column_names)
            
            val_encoding=val_ds.map(preprocess_data,  
                                           fn_kwargs={'tokenizer': tokenizer,'labels_list':labels_list,'label2id':label2id,'k_tok':args.corpus},
                                           batched=True,
                                           remove_columns=val_ds.column_names)
            model = AutoModelForSequenceClassification.from_pretrained(bert_model_loc, 
                                                                   problem_type=classification_type, 
                                                                   num_labels=len(labels_list),
                                                                   id2label=id2label,
                                                                   label2id=label2id
                                                                   )

            targs = TrainingArguments(
                output_dir=model_part_dir_loc,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1_micro",
                greater_is_better=True,
                num_train_epochs=20,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=2e-5,
                weight_decay=0.01,
                save_safetensors=False
            )


            trainer = Trainer(
                model=model,
                args=targs,
                train_dataset=train_encoding,
                eval_dataset=test_encoding,           # 10 % validation set
                tokenizer=tokenizer,
                compute_metrics=lambda p: hf_compute_metrics(p, id2label),
            )
            trainer.train()
            #evaluation_results = trainer.evaluate()

            # ---------------------------------------------
            # 1) pick best checkpoint (already done by HF)
            # 2) run on the original dev = test set
            # ---------------------------------------------
            test_pred = trainer.predict(test_encoding)    # EvalPrediction

            # sigmoid logits -> binary predictions for metrics
           
            if isinstance(test_pred.predictions, tuple):
                test_logits = test_pred.predictions[0]
            else:
                test_logits = test_pred.predictions
            test_result= multi_label_metrics(test_logits, test_pred.label_ids, id2label)
            print("Test metrics:", test_result)

            # ---------------------------------------------
            # 3) save for paired bootstrap (needs gold labels)
            # ---------------------------------------------
            out_pickle = os.path.join(result_part_dir_loc,
                          f"results_seed-{args.seed}.pkl")
            with open(out_pickle, "wb") as f:
                pickle.dump({
                    "seed":    args.seed,
                    "metrics": test_result,
                    "preds":   test_logits.tolist(),
                    "labels":  test_pred.label_ids.tolist(),
                    "doc_ids": test_encoding["doc_id"],
                }, f, protocol=pickle.HIGHEST_PROTOCOL)