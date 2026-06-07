# DyGMA

This repository contains the implementation of **DyGMA** for document-level event argument extraction.

DyGMA is built on a multi-event prompt extraction framework and adds dynamic graph-memory reasoning, including dependency-aware role planning, dual-stream graph aggregation, and memory-enhanced role queries.

## Setup

Create an environment and install dependencies:

```bash
conda create -n dygma python=3.9
conda activate dygma
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The code uses RoBERTa as the backbone. Set the path or Hugging Face identifier with `MODEL_NAME_OR_PATH` when running scripts.

Coreference features are optional. If you use fastcoref, download the model separately and pass its local path with `--coref_model_path`, or set:

```bash
export DYGMA_COREF_MODEL=/path/to/fcoref_model
```

## Data

Dataset files are not included in this repository. Prepare RAMS, WikiEvents, and MLEE following their original licenses and place processed files under:

```text
data/RAMS_1.0/data_final/
data/WikiEvent/data_final/
data/MLEE/data_final/
```

Prompt and schema description files are provided under `data/prompts/` and `data/dset_meta/`.

## Training

Run the scripts below after preparing data and the backbone model:

```bash
MODEL_NAME_OR_PATH=roberta-large bash scripts/train_wikievent_roberta.sh
MODEL_NAME_OR_PATH=roberta-large bash scripts/train_rams_roberta.sh
MODEL_NAME_OR_PATH=roberta-large bash scripts/train_mlee_roberta.sh
```

Useful environment variables:

```bash
CUDA_DEVICE=0
SEEDS="22 42 66 99 111 1234"
LR=3e-5
```

## Inference

Use `--inference_only` with a trained checkpoint:

```bash
python -u engine.py \
  --model_type=DyGMA \
  --dataset_type=wikievent \
  --model_name_or_path=roberta-large \
  --inference_only \
  --inference_model_path=/path/to/checkpoint \
  --role_path=./data/dset_meta/description_wikievent.csv \
  --prompt_path=./data/prompts/prompts_wikievent_full.csv
```

## Citation

Citation information will be added after publication.
