# ICD-9 Automated Clinical Coding

This directory contains the scripts and configurations to run the ICD-9 automated coding task using the K-Tokenizer framework.

## Prerequisites

* Python 3.7+

* GPU with CUDA (optional, recommended for speed)

* Install dependencies:

  ```bash
  pip install -r ../requirements.txt
  ```

* Ensure your environment variable points to the correct GPU (optional):

  ```bash
  export CUDA_VISIBLE_DEVICES=1
  ```

* To load the dataset, follow [PLM-ICD](https://github.com/MiuLab/PLM-ICD) or contact [abul.hasan@phc.ox.ac.uk](mailto:abul.hasan@phc.ox.ac.uk)

## Script: `run_icd_ktok.py`

This script fine-tunes a clinical BERT model for ICD-9 code prediction on the MIMIC-III dataset.

### Usage

From this folder, execute:

```bash
export CUDA_VISIBLE_DEVICES=1
python3 run_icd_ktok.py \
  --code_50 \
  --train_file ../data/mimic3/train_50.csv \
  --validation_file ../data/mimic3/dev_50.csv \
  --max_length 512 \
  --chunk_size 128 \
  --model_name_or_path abulhasan/clinical-bert-ktokenizer \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 1 \
  --num_train_epochs 20 \
  --num_warmup_steps 2000 \
  --output_dir ../models/ClinicalKBERT_UMLS_20_Code_50/ \
  --model_type bert \
  --model_mode laat \
  --result_path ../results/ClinicalKBERT_UMLS_20_Code_50/ \
  --corpus umls \
  --tokenizer_base_loc ../umls_tokenization/ \
  --seed 2048 \
  --fertility 0.0 \
  --percent_data 5
```

### Arguments

| Flag                            | Description                                       |
| ------------------------------- | ------------------------------------------------- |
| `--code_50`                     | Enable top-50 ICD-9 code prediction               |
| `--train_file`                  | Path to training CSV file                         |
| `--validation_file`             | Path to validation CSV file                       |
| `--max_length`                  | Maximum token length for inputs                   |
| `--chunk_size`                  | Window size for chunked inputs                    |
| `--model_name_or_path`          | Hugging Face model identifier or local path       |
| `--per_device_train_batch_size` | Batch size per GPU for training                   |
| `--gradient_accumulation_steps` | Number of gradient accumulation steps             |
| `--per_device_eval_batch_size`  | Batch size per GPU for evaluation                 |
| `--num_train_epochs`            | Number of fine-tuning epochs                      |
| `--num_warmup_steps`            | Linear scheduler warmup steps                     |
| `--output_dir`                  | Directory to save model checkpoints               |
| `--model_type`                  | Model architecture (e.g., `bert`)                 |
| `--model_mode`                  | Model variant or task mode (e.g., `laat`)         |
| `--result_path`                 | Directory to write prediction results             |
| `--corpus`                      | Corpus variant (`base`, `umls`, or `mimic`)       |
| `--tokenizer_base_loc`          | Local path to tokenizer resources                 |
| `--seed`                        | Random seed for reproducibility                   |
| `--fertility`                   | Fertility hyperparameter for coding               |
| `--percent_data`                | Percentage of training data to use (for ablation) |

---

Ensure directories `../models/` and `../results/` exist or will be created by the script. Adjust flags as needed for alternative ICD code sets or data splits. Good luck!
