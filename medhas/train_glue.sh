#!/bin/sh
export GLUE_DIR=/home/keyur/medhas/glue_data/
#export MODEL_NAME=bert-base-cased
#export MODEL_NAME=albert-base-v2
export MODEL_NAME=distilbert-base-cased
export DATESTAMP=20200731
export EXPERIMENT_DIR=/mnt/data/medhas/glue_experiments
mkdir -p $EXPERIMENT_DIR/$MODEL_NAME/$DATESTAMP

#for task in CoLA STS-B MRPC RTE WNLI SST-2 QQP MNLI QNLI
for task in CoLA
do
  export TASK_NAME=$task
  export OUTPUT_DIR=$EXPERIMENT_DIR/$MODEL_NAME/$DATESTAMP/$TASK_NAME
  export LOG_DIR=$EXPERIMENT_DIR/$MODEL_NAME/$DATESTAMP/$TASK_NAME/logs
  echo $TASK_NAME

  logging_steps=500
  if [ "$task" = "RTE" ]; then
    logging_steps=10
  elif [ "$task" = "WNLI" ]; then
    logging_steps=10
  elif [ "$task" = "MRPC" ]; then
    logging_steps=25
  elif [ "$task" = "CoLA" ]; then
    logging_steps=50
  elif [ "$task" = "STS-B" ]; then
    logging_steps=25
  elif [ "$task" = "SST-2" ]; then
    logging_steps=200
  elif [ "$task" = "QNLI" ]; then
    logging_steps=250
  elif [ "$task" = "QQP" ]; then
    logging_steps=1000
  elif [ "$task" = "MNLI" ]; then
    logging_steps=1000
  fi

  python ../examples/text-classification/run_glue.py  \
  --model_name_or_path $MODEL_NAME  \
  --task_name $TASK_NAME  \
  --do_train  --do_eval  \
  --data_dir $GLUE_DIR/$TASK_NAME  \
  --max_seq_length 128   \
  --per_device_train_batch_size 32   \
  --per_device_eval_batch_size 32   \
  --learning_rate 2e-5   \
  --num_train_epochs 3.0   \
  --output_dir $OUTPUT_DIR  \
  --logging_dir $LOG_DIR \
  --logging_steps $logging_steps \
  --evaluate_during_training \
  --save_total_limit 2 \
  --save_steps 1000 \
  --gradient_accumulation_steps 1 \
  --overwrite_output_dir
  
  #sleep 600s
done

