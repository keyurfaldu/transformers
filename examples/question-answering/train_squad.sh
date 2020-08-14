#!/bin/sh
export DATA_DIR=/home/keyur/medhas/hf211/squad_data
export MODEL_NAME=bert-base-cased
export DATESTAMP=20200724
export EXPERIMENT_DIR=/mnt/data/medhas/squad_experiments
mkdir -p $EXPERIMENT_DIR/$MODEL_NAME/$DATESTAMP

python run_squad.py \
	--model_type bert \
	--model_name_or_path bert-base-uncased \
	--do_train \
	--do_eval \
	--do_lower_case \
	--train_file $DATA_DIR/train-v1.1.json \
	--predict_file $DATA_DIR/dev-v1.1.json \
	--per_gpu_train_batch_size 16 \
	--learning_rate 3e-5 \
	--num_train_epochs 2.0 \
	--max_seq_length 384 \
	--doc_stride 128 \
	--output_dir $EXPERIMENT_DIR/$MODEL_NAME/$DATESTAMP \
	--threads 12

