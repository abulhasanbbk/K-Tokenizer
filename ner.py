#!/usr/bin/env python
# coding=utf-8
"""
Author: Abul Hasan
Automated ner module which will take care of alpha parameter and fertilites for generating results
It will always run for 4 fertilites : [0, 0.035, 0.065, 1] 
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
    # Initialize lists to store sentences and corresponding tags
    sent_docs = []  # a list of sentences
    tag_docs = []   # a list of corresponding tags
    
    # Open the file at the specified location
    with open(location, "r") as f:
        lines = f.readlines()
        sent = []  # to store the current sentence
        tags = []  # to store the corresponding tags
        
        # Iterate through each line in the file
        for line in lines:
            # Stripping off the new line character
            if line[-1:] == '\n':
                line = line.strip()
            
            # Split the line into words
            line = line.split()
            
            # Try to populate the sentence and tags lists, exception occurs at the boundary of a sentence
            try:
                sent.append(line[word_col])
                tags.append(line[tag_col])
            except:
                # Append the collected sentence and tags to the respective lists
                sent_docs.append(sent)
                tag_docs.append(tags)
                
                # Reset the sentence and tags lists for the next sentence
                sent = []
                tags = []
                
                # Move to the next line
                continue
        
        # Return the collected sentence and tag lists
    return sent_docs, tag_docs

#Creating huggingface dataset object by providing two features
def create_hugging_face_dataset(sents, tags, tag2id):
    numeric_tags = []  # List to store numeric tags

    # Convert tags to numeric
    for tag in tags:
        num_tag = [tag2id[t] for t in tag]
        numeric_tags.append(num_tag)
    
    # Create a dataset dictionary with "tokens" and "ner_tags" columns
    datasetDict = {"tokens": sents, "ner_tags": numeric_tags}
    
    # Create a Dataset object by loading the dictionary
    raw_datasets = Dataset.from_dict(datasetDict)
    
    return raw_datasets

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
    #tokenized_inputs["prediction_mask"] = pred_msks
    
    return tokenized_inputs




def compute_metrics(p, label_list,result_loc, mt=[0]):
    predictions, labels = p
    
    # Convert raw predictions to label sequences
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                        for prediction, label in zip(predictions, labels)]
    
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                   for prediction, label in zip(predictions, labels)]
    
    # Compute evaluation metrics using seqeval
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    print("*****Leninent*******************")
    print(classification_report(true_labels, true_predictions))
    print("****************Strict***************")
    print(classification_report(true_labels, true_predictions, mode='strict', scheme=IOB2))
    
    mt[0] += 1  # Increment mutable variable (evaluated only once)
    print(mt)
    print(results)
    file_name='result_'+ str(mt[0])+'.pkl'
    file_path = os.path.join(result_loc, file_name)
    # Save results for each epoch to a pickle file
    with open(file_path, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Return computed metrics
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }

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
    train_sents, train_tags=load_n2c2_data(train_data_loc)
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
    test_sents, test_tags=load_n2c2_data(test_data_loc)
    test_dataset=create_hugging_face_dataset(test_sents, test_tags, label2id)
    test_dataset_tokenized= test_dataset.map(
        lambda examples: tokenize_and_align_labels_pred_msk(examples, tokenizer,corpus),
        batched=True,
        remove_columns=["tokens", "ner_tags"]
        )
    for idx, part in enumerate(partitions_sents):
        training_sents=part
        training_tags=partitions_tags[idx]
        train_dataset=create_hugging_face_dataset(training_sents, training_tags, label2id)
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
            num_train_epochs=10,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset_tokenized,
            eval_dataset=test_dataset_tokenized,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda p: compute_metrics(p, label_list, result_part_dir_loc),
        )

        trainer.train()

   

if __name__ == "__main__":
   
   parser = argparse.ArgumentParser()
   # parse arguments
   parser.add_argument("--model", type=str, required=True,
                        help="Provide model name: clinicalBERT, BioBERT ")
   parser.add_argument("--type", type=str, required=True,
                        help="Provide type of the initialization: infuse, random ")
   parser.add_argument("--tokenizer_base_loc", type=str, required=True,
                        help="Provide locations where all your tokenizer file resides ")
   parser.add_argument("--corpus", type=str, required=True,
                        help="Provide corpus type:umls or mimic ")
   parser.add_argument("--alpha", type=int, required=True,
                        help="Provide alpha parameter[20-30] of K-Tokenizer")
   parser.add_argument("--partition", type=int, required=True,
                        help="Provide partition of the training dataset, 1 for 100%")
   parser.add_argument("--seed", type=int, required=True,
                        help="Provide 42, 2046, or 1234")
   parser.add_argument("--numrun", type=int, required=True,
                        help="Provide run number, 1,2,or 3")
   args = parser.parse_args()
   set_seed(args.seed)
   if args.corpus=="umls":
       if args.model=="clinicalBERT" and args.type=='infuse':
           fileNames=ClinicalBERTFromUMLSFileNames()
           infusedBertModelFileName=InfusedClinicalBertFromUMLSFileNames()
   elif args.corpus=="mimic":
       if args.model=="clinicalBERT" and args.type=='infuse':
           fileNames=ClinicalBERTFromMIMICFileNames()
           infusedBertModelFileName=InfusedClinicalBertFromMIMICFileNames()  
   else:
       pass
   #Creating locations for K-tokenizer depending on their alpha values
   alpha=args.alpha
   deltas=[0]
   
   for delta in deltas:
       if args.corpus=="umls":
           locations=Loc()
           locations.base_dir=args.tokenizer_base_loc
           locations.base=fileNames.base_tok
           locations.disease=os.path.join(args.tokenizer_base_loc, fileNames.base_loc_disease)
           locations.drug=os.path.join(args.tokenizer_base_loc, fileNames.base_loc_drug)
           locations.ktok=os.path.join(args.tokenizer_base_loc, fileNames.base_loc_ktok)
       elif args.corpus=="mimic":
           locations=Loc()
           locations.base_dir=args.tokenizer_base_loc
           locations.base=fileNames.base_tok
           locations.disease=os.path.join(args.tokenizer_base_loc, fileNames.base_loc_disease+str(alpha)+"-"+str(delta))
           locations.drug=os.path.join(args.tokenizer_base_loc, fileNames.base_loc_drug+str(alpha)+"-"+str(delta))
           locations.ktok=os.path.join(args.tokenizer_base_loc, fileNames.base_loc_ktok+str(alpha)+"-"+str(delta))
       elif args.corpus=="base":
           locations="emilyalsentzer/Bio_ClinicalBERT"
           pass
       
       #defining fertilities for the K-Tokenizer
       fertilities=[0]#, 0.035, 0.065, 1]
       #fertilities=[0]
       #create base directory for results
       result_main_loc="n2c2/Results-3-Runs-v2/"
       check_or_create_folder(result_main_loc)
       result_main_loc=os.path.join(result_main_loc, "Run_"+str(args.numrun))
       check_or_create_folder(result_main_loc)
       model_main_loc="n2c2/Models-3-Runs/"
       check_or_create_folder(model_main_loc)
       #model_main_loc=os.path.join(model_main_loc, "Run_"+str(args.numrun))
       check_or_create_folder(model_main_loc)
       if args.corpus=="base": 
           result_base_loc=os.path.join(result_main_loc, args.model+"_"+args.corpus)
           model_base_loc=os.path.join(model_main_loc, args.model+"_"+args.corpus)
       else:
           result_base_loc=os.path.join(result_main_loc, "K_"+args.model+"_"+args.corpus)
           model_base_loc=os.path.join(model_main_loc, "K_"+args.model+"_"+args.corpus)

         

       train_data_loc="/working/abul/K-Tokenizer/n2c2/data/dataset1_train.txt"
       test_data_loc="/working/abul/K-Tokenizer/n2c2/data/dataset1_test.txt"
       for f in fertilities:
           print("***********************************************")
           print("Running NER for fertility: ", f)
           print("Model Name: ", args.model)
           print("Model Type: ", args.type)
           print("Corpus Type: ", args.corpus)
           print("Alpha Parameter: ", args.alpha)
           print("Delta Parameter: ", delta)
           if args.partition==1:
               part="100"
           elif args.partition==2:
               part="50"
           elif args.partition==3:
               part="30"
           elif args.partition==5:
               part="20"
           else:
               print("Please provide a valid partition size for Training data: 1,2,3,5 for 100%, 50%, 30%, and 20 respectively")
           result_dir=args.model+"_"+args.type+"_"+args.corpus+"_"+"alpha_"+str(args.alpha)+"_"+"delta_"+str(delta)+"_fertility_"+str(f*1000)+"_"+part+"_"+"percent"
           result_loc=os.path.join(result_base_loc, result_dir)
           print(result_loc)
           check_or_create_folder(result_loc)
           model_loc=os.path.join(model_base_loc, result_dir)
           check_or_create_folder(model_loc) 
           print("***********************************************")
           
           if args.corpus=="umls":
               old_loc="/working/abul/two_stage_fine_tunning/"
               bert_model_loc=os.path.join(old_loc, infusedBertModelFileName.infused_bert_model) #using the 13000 ktokenizer files
               #print("********************", bert_model_loc)
               #+time.sleep(20)
               tokenizer=KTokenizerFromUMLS(locations, f)
               run_ner(tokenizer, bert_model_loc, result_loc, model_loc,train_data_loc, test_data_loc, args.partition, args.corpus)
           elif args.corpus=="mimic":
               bert_model_loc=infusedBertModelFileName.infused_bert_model+str(alpha)+"-"+str(delta)
               tokenizer=KTokenizerFromMIMICIII(locations, f)
               run_ner(tokenizer, bert_model_loc, result_loc,model_loc, train_data_loc, test_data_loc, args.partition,args.corpus)
           elif args.corpus=="base":
                bert_model_loc=locations
                tokenizer=AutoTokenizer.from_pretrained(locations)
                run_ner(tokenizer, bert_model_loc, result_loc, model_loc, train_data_loc, test_data_loc, args.partition,args.corpus)

           else:
               pass

    
   