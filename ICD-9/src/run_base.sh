export CUDA_VISIBLE_DEVICES=1
python3 run_icd.py \
 --code_50 \
 --train_file ../data/mimic3/train_50.csv \
 --validation_file ../data/mimic3/dev_50.csv \
 --code_file ../data/mimic3/ALL_CODES_50.txt \
 --max_length 512 \
 --chunk_size 128 \
 --model_name_or_path emilyalsentzer/Bio_ClinicalBERT \
 --per_device_train_batch_size 1 \
 --gradient_accumulation_steps 8 \
 --per_device_eval_batch_size 1 \
 --num_train_epochs 20 \
 --num_warmup_steps 2000 \
 --output_dir ../models/clinicalBERT_model_100_Code_50/ \
 --model_type bert \
 --model_mode laat \
 --seed 2048 \
#test 
python3 run_icd.py \
 --code_50 \
 --train_file ../data/mimic3/train_50.csv \
 --validation_file ../data/mimic3/dev_50.csv \
 --code_file ../data/mimic3/ALL_CODES_50.txt \
 --max_length 512 \
 --chunk_size 128 \
 --model_name_or_path emilyalsentzer/Bio_ClinicalBERT \
 --per_device_train_batch_size 1 \
 --gradient_accumulation_steps 8 \
 --per_device_eval_batch_size 1 \
 --num_train_epochs 0 \
 --num_warmup_steps 2000 \
 --output_dir ../models/clinicalBERT_model_100_Code_50/ \
 --model_type bert \
 --model_mode laat \
 --seed 2048 \
 --result_save ../results/clinicalBERT_model_100_Code_50/


