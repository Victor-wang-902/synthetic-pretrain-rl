# Having obtain the datasets, you can pre-train the model with the following command
accelerate launch --config_file /path/to/hf/accelerate/config/ pretrain/pretrain_dist.py --dataset /path/to/dataset.csv --batch_size 32768 --embed_dim 128 --n_layer 3 --n_head 1 --outdir /path/to/checkpoint/folder
