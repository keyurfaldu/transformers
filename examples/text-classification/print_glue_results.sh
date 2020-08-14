#!/bin/sh
export MODEL_NAME=albert-base-v2
export DATESTAMP=20200715
export EXPERIMENT_DIR=/mnt/data/medhas/glue_experiments

echo $MODEL_NAME
for task in  CoLA SST-2 MRPC STS-B QQP MNLI QNLI RTE WNLI
do
  echo $task
  cat $EXPERIMENT_DIR/$MODEL_NAME/$DATESTAMP/$task/eval_results*
done
