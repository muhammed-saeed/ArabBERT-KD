# Knowledge Distillation of BERT Language Model on the Arabic Language


## Installation

### Requirements

- Python 3.8.16

### Environment

1. Create a virtual environment and activate it.

python3 -m venv env
source env/bin/activate

### 2. Install transformers .
we have cloned the transformer library and then modified the library to work with the arabic.

The orignal transformers library link  

`https://github.com/huggingface/transformers` 

but the `distillation` folder has been modified to work with the arabic models

```
cd transformes
pip install .
pip install -r transformers/examples/research_projects/distillation/requirements.txt
```



## Training

### binarizating Data
```
python scripts/binarized_data.py \
    --file_path arabic.txt \
    --tokenizer_type bert \
    --tokenizer_name asafaya/bert-large-arabic \
    --dump_file data/binarized_text
```

### Token counts
```
python scripts/token_counts.py \
    --data_file data/binarized_text.pickle \
    --token_counts_dump data/token_counts.pickle \
    --vocab_size 32000
```
### Train.py
using single GPU
```
python train.py \
    --student_type distilbert \
    --student_config training_configs/distilbert-base-uncased.json \
    --teacher_type bert \
    --teacher_name asafaya/bert-large-arabic \
    --alpha_ce 5.0 --alpha_mlm 2.0 --alpha_cos 1.0 --alpha_clm 0.0 --mlm \
    --freeze_pos_embs \
    --dump_path distilbert_output/serialization_dir/my_first_training \
    --data_file data/binarized_text.pickle \
    --token_counts data/token_counts.pickle \
    --checkpoint_interval 100000 \
    --force # overwrites the `dump_path` if it already exists.
```

Using multi GPU

```
export CUDA_VISIBLE_DEVICES=1,2

export NODE_RANK=0
export N_NODES=1

export N_GPU_NODE=2 # (CHANGE THIS ACCORDING TO YOUR GPUS)
export WORLD_SIZE=2 # (CHANGE THIS ACCORDING TO YOUR GPUS)
export MASTER_PORT=42562 # (you can use 29500)
export MASTER_ADDR="localhost" # (leave as it is)


pkill -f 'python -u  transformers/examples/research_projects/distillation/train.py' # (CHANGE DEPENDING ON YOUR PATH)

python -m torch.distributed.launch \
    --nproc_per_node=$N_GPU_NODE \
    --nnodes=$N_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    transformers/examples/research_projects/distillation/train.py 
        --n_gpu $WORLD_SIZE \
        --student_type distilbert \
        --student_config transformers_/transformers/examples/research_projects/distillation/training_configs/distilbert-base-uncased.json 
        --student_pretrained_weights transformers/examples/research_projects/distillation/The_data/tf_bert-base-uncased_0247911.pth 
        --teacher_type bert \
        --teacher_name asafaya/bert-large-arabic \
        --alpha_ce 5.0 --alpha_mlm 2.0 --alpha_cos 1.0 --alpha_clm 0.0 --mlm \
        --freeze_pos_embs \
        --dump_path Dumps/ 
        --data_file transformers/examples/research_projects/distillation/The_data/merged_data_binarized.pickle 
        --token_counts transformers/examples/research_projects/distillation/The_data/merged_token_count.pickle 
        --batch_size 4 
        --learning_rate 3e-5 \
        --checkpoint_interval 100000 \
        --force 
```
