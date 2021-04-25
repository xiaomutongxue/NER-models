CURRENT_DIR=`pwd`
export GLUE_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="cner"
export CUDA_VISIBLE_DEVICES="0"

python3 run_birnn_crf.py \
  --task_name=$TASK_NAME \
  --do_train \
  --do_predict \
  --data_dir=$GLUE_DIR/${TASK_NAME}/ \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_birnn_crf/ \
  --max_seq_len=128 \
  --num_epoch=1 \
  --batch_size=32 \
  --save_best_val_model \
  --recovery \
  --seed=42