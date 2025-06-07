#!/usr/bin/env python
# coding=utf-8
"""
Author: Abul Hasan
NER module for n2c2 data. 
"""
import argparse
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed  
)
from datasets import Dataset
import evaluate
import numpy as np
import pickle
import os
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from utils import *
import time
seqeval = evaluate.load("seqeval")
RESULT_BASE_LOC="Results"
MODEL_SAVE_LOC="Saved_Models"
INFUSED_UMLS_MODEL_BASE_LOC="abulhasan/clinical-bert-ktokenizer"
INFUSED_MIMIC_MODEL_BASE_LOC="../infused_mimic_clinical_bert"
NUM_PARTITION=1
N2C2_DATA_LOC="n2c2_data"
TRAIN_FILE_NAME="train.txt"
TEST_FILE_NAME="test.txt"
def load_from_pickle(file_path):
    """
    Load data from a pickle file.

    Parameters:
        file_path (str): The file path from which the data will be loaded.

    Returns:
        any: The loaded data.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print("Folder created:", folder_path)
    else:
        print("Folder already exists:", folder_path)

def load_n2c2_data(location, word_col=7, tag_col=8):
    """
    Returns
        sents : list[list[str]]
        tags  : list[list[str]]
        ids   : list[str]      (fileName::sentIdx => stable, cross-run)
    """
    sents, tags_all, ids = [], [], []
    sent, tags = [], []
    prev_path, sent_idx = None, 0

    with open(location, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            cols = line.split()

            # sentence boundary in the TSV  (empty line / try-except in old code)
            if len(cols) < max(word_col, tag_col) + 1:
                if sent:
                    ids.append(f"{prev_path}::{sent_idx}")
                    sent_idx += 1
                    sents.append(sent);  tags_all.append(tags)
                    sent, tags = [], []
                continue

            doc_path = cols[0]                # “data/test/116903.txt”
            if prev_path is None:
                prev_path = doc_path
            if doc_path != prev_path:         # new document → reset idx
                prev_path = doc_path
                sent_idx = 0

            sent.append(cols[word_col])
            tags.append(cols[tag_col])

        # flush last sentence
        if sent:
            ids.append(f"{prev_path}::{sent_idx}")
            sents.append(sent); tags_all.append(tags)

    return sents, tags_all, ids


#Creating huggingface dataset object by providing two features
def create_hugging_face_dataset(sents, tags, sent_ids, tag2id):
    numeric_tags = [[tag2id[t] for t in seq] for seq in tags]
    ds_dict = {"tokens": sents,
               "ner_tags": numeric_tags,
               "sid": sent_ids}            # <-- keep IDs in dataset rows
    return Dataset.from_dict(ds_dict)

def align_labels_pred_msk_with_tokens(labels, word_ids):
    new_labels = []  # List to store new labels
    pred_msk = []     # List to store prediction masks
    label_all_tokens = True  # Flag to determine whether to label all tokens

    previous_word_idx = None   # Initialize previous_word_idx to None

    # Iterate through word_ids and align labels and prediction masks
    for word_idx in word_ids:
        # For special tokens with None word_id, set label to -100 and pred_msk to False
        if word_idx is None:
            new_labels.append(-100)
            pred_msk.append(False)
        elif word_idx != previous_word_idx:
            # If not a repeated token, set label to either labels[word_idx] or -100 based on label_all_tokens
            new_labels.append(labels[word_idx] if label_all_tokens else -100)
            pred_msk.append(True)
        else:
            # If repeated token, set label to either labels[word_idx] or -100 based on label_all_tokens
            new_labels.append(labels[word_idx] if label_all_tokens else -100)
            pred_msk.append(True if label_all_tokens else False)
        
        previous_word_idx = word_idx  # Update previous_word_idx for the next iteration

    return new_labels, pred_msk

def tokenize_and_align_labels_pred_msk(examples, tokenizer, corpus):
    # Tokenize the input sentences and align labels and prediction masks
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    
    all_labels = examples["ner_tags"]  # List of all label sequences
    new_labels = []  # List to store aligned labels for all sequences
    #pred_msks = []  # List to store prediction masks for all sequences
    
    # Iterate through label sequences and align labels and prediction masks
    for i, labels in enumerate(all_labels):
        # Word IDs for the current sequence
        if corpus=="umls" or corpus=="mimic":
            word_ids = tokenized_inputs["word_ids"][i]
        elif corpus=="base":     
            word_ids=tokenized_inputs.word_ids(i)
        else:
            pass
        aligned_labels, pred_mask = align_labels_pred_msk_with_tokens(labels, word_ids)
        new_labels.append(aligned_labels)
        #pred_msks.append(pred_mask)
    
    # Add aligned labels and prediction masks to the tokenized inputs
    tokenized_inputs["labels"] = new_labels
    tokenized_inputs["sids"] = examples["sid"]
    #tokenized_inputs["prediction_mask"] = pred_msks
    
    return tokenized_inputs

def compute_metrics(p,
                    label_list,
                    result_loc,
                    sid_list,          # <-- plain list of sentence IDs
                    mt=[0],
                    seed=None):
    preds, labels = p
    preds = np.argmax(preds, axis=2)

    true_preds  = [[label_list[p] for p,l in zip(pr,la) if l!=-100]
                   for pr,la in zip(preds, labels)]
    true_labels = [[label_list[l] for p,l in zip(pr,la) if l!=-100]
                   for pr,la in zip(preds, labels)]

    metrics = seqeval.compute(predictions=true_preds,
                              references=true_labels)

    mt[0] += 1
    fname = f"results_epoch-{mt[0]}_seed-{seed}.pkl"
    with open(os.path.join(result_loc, fname), "wb") as f:
        pickle.dump({
            "epoch":   mt[0],
            "seed":    seed,
            "metrics": metrics,
            "preds":   true_preds,
            "labels":  true_labels,
            "sent_ids": sid_list          # <- save the full ordered list
        }, f)
    return {"precision": metrics["overall_precision"],
            "recall":    metrics["overall_recall"],
            "f1":        metrics["overall_f1"],
            "accuracy":  metrics["overall_accuracy"]}

def run_ner(tokenizer, bert_model_loc, result_loc, model_loc,train_data_loc, test_data_loc, partition, corpus):
    #out_dir="bert_models"
    #check_or_create_folder(out_dir)
    print("*********************************************************")
    print("Running NER module for: ")
    print(bert_model_loc)
    print(result_loc)
    print(train_data_loc)
    print(test_data_loc)
    print("**********************************************************")
    train_sents, train_tags,train_ids=load_n2c2_data(train_data_loc)
    unique_tags = set(tag for doc in train_tags for tag in doc)
    label_list=list(unique_tags)
    print(label_list)
    id2label={}
    for idx, label in enumerate(label_list):
        id2label[idx]=label
    label2id={}
    for idx,label in enumerate(label_list):
        label2id[label]=idx
    #crearting partition size
    percentage=partition #10%,20%,30%,40%,50% of the dataset
    partition_size = len(train_sents) // percentage
    # Divide train_sents and train_tags into 10 equal parts
    partitions_sents = [train_sents[i:i+partition_size] for i in range(0, len(train_sents), partition_size)]
    partitions_tags = [train_tags[i:i+partition_size] for i in range(0, len(train_tags), partition_size)]
    partition_sent_ids= [train_ids[i:i+partition_size] for i in range(0, len(train_ids), partition_size)]
    test_sents, test_tags, test_ids=load_n2c2_data(test_data_loc)
    test_dataset=create_hugging_face_dataset(test_sents, test_tags, test_ids, label2id)
    test_dataset_tokenized= test_dataset.map(
        lambda examples: tokenize_and_align_labels_pred_msk(examples, tokenizer,corpus),
        batched=True,
        remove_columns=["tokens", "ner_tags"]
        )
    for idx, part in enumerate(partitions_sents):
        training_sents=part
        training_tags=partitions_tags[idx]
        training_ids=partition_sent_ids[idx]
        train_dataset=create_hugging_face_dataset(training_sents, training_tags, training_ids, label2id)
        print(train_dataset[0])

        #tokenizer = AutoTokenizer.from_pretrained("/working/abul/lm/BERTfromScratch/models/bert-bpe-cased")
       
        train_dataset_tokenized = train_dataset.map(
            lambda examples: tokenize_and_align_labels_pred_msk(examples, tokenizer, corpus),
            batched=True,
            remove_columns=["tokens", "ner_tags"]
        )

        
        #print(train_dataset_tokenized[0])
    
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
   

        model = AutoModelForTokenClassification.from_pretrained(bert_model_loc, 
                                                                num_labels=len(unique_tags), 
                                                                id2label=id2label, 
                                                                label2id=label2id)
        #for param in model.parameters():
        #    param.requires_grad = False

        # Define which parameters you want to train
        # For example, if you want to train the classifier layer:
        #for param in model.classifier.parameters():
        #    param.requires_grad = True

        part_dir="part_"+str(idx)
        
        out_part_loc=os.path.join(model_loc, part_dir)
        check_or_create_folder(out_part_loc)

        result_part_dir_loc=os.path.join(result_loc, part_dir)
        check_or_create_folder(result_part_dir_loc)
        #create part directories

        training_args = TrainingArguments(
            output_dir=out_part_loc,
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,                 # <-- only 3 epochs
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            push_to_hub=False,
            save_total_limit=3                  # keep only 3 checkpoints
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset_tokenized,
            eval_dataset=test_dataset_tokenized,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda p: compute_metrics(p, 
                                                      label_list, 
                                                      result_part_dir_loc,
                                                      sid_list=test_ids,           # <<< here
                                                      seed=args.seed),
        )
        trainer.train()

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True, help="Provide corpus type:umls or mimic ")
    parser.add_argument("--seed", type=int, required=True,  help="Provide 42, 2046, or 1234")
    args = parser.parse_args()
    set_seed(args.seed)
    fertilities=[0, 0.035,0.065,1.0]
    finetuned_save_loc=args.corpus+"_fine_tuned_model_"+str(args.seed)
    if args.corpus=="umls":
        fileNames=ClinicalBERTFromUMLSFileNames()
        infusedBertModelFileName=InfusedClinicalBertFromUMLSFileNames()
        locations=Loc()
        tokenizer_base_dir=args.corpus+"_tokenization"
        locations.base_dir=tokenizer_base_dir
        locations.base=fileNames.base_tok
        locations.disease=os.path.join(tokenizer_base_dir, fileNames.base_loc_disease)
        locations.drug=os.path.join(tokenizer_base_dir, fileNames.base_loc_drug)
        locations.ktok=os.path.join(tokenizer_base_dir, fileNames.base_loc_ktok)
        
    elif args.corpus=="mimic":
        fileNames=ClinicalBERTFromMIMICFileNames()
        infusedBertModelFileName=InfusedClinicalBertFromMIMICFileNames()
        tokenizer_base_dir=args.corpus+"_tokenization"
        locations=Loc()
        locations.base_dir=tokenizer_base_dir
        locations.base=fileNames.base_tok
        locations.disease=os.path.join(tokenizer_base_dir, fileNames.base_loc_disease)
        locations.drug=os.path.join(tokenizer_base_dir, fileNames.base_loc_drug)
        locations.ktok=os.path.join(tokenizer_base_dir, fileNames.base_loc_ktok)
        
    elif args.corpus=="base":
        locations="emilyalsentzer/Bio_ClinicalBERT"
        pass
    
    train_data_loc=os.path.join(N2C2_DATA_LOC, TRAIN_FILE_NAME)
    test_data_loc=os.path.join(N2C2_DATA_LOC, TEST_FILE_NAME)
    result_dir="results_"+args.corpus+"_"+str(args.seed)
    check_or_create_folder(RESULT_BASE_LOC)
    result_loc=os.path.join(RESULT_BASE_LOC, result_dir)
    check_or_create_folder(result_loc)
    check_or_create_folder(MODEL_SAVE_LOC)
    fine_tuned_model_loc=os.path.join(MODEL_SAVE_LOC, finetuned_save_loc)
    check_or_create_folder(fine_tuned_model_loc) 
    if args.corpus=="umls":
        bert_model_loc=INFUSED_UMLS_MODEL_BASE_LOC #using huggingface files
        f=fertilities[0]
        tokenizer=KTokenizerFromUMLS(locations, f)
        run_ner(tokenizer, bert_model_loc, result_loc, fine_tuned_model_loc,train_data_loc, test_data_loc, NUM_PARTITION, args.corpus)
    elif args.corpus=="mimic":
        bert_model_loc=os.path.join(INFUSED_MIMIC_MODEL_BASE_LOC, infusedBertModelFileName.infused_bert_model) #using the 13000 ktokenizer files
        f=fertilities[0]
        tokenizer=KTokenizerFromMIMICIII(locations, f)
        run_ner(tokenizer, bert_model_loc, result_loc, fine_tuned_model_loc,train_data_loc, test_data_loc, NUM_PARTITION, args.corpus)
    elif args.corpus=="base":
        bert_model_loc=locations
        tokenizer=AutoTokenizer.from_pretrained(locations)
        run_ner(tokenizer, bert_model_loc, result_loc, fine_tuned_model_loc,train_data_loc, test_data_loc, NUM_PARTITION, args.corpus)
    else:
        pass
   