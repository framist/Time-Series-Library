export CUDA_VISIBLE_DEVICES=0

model_name=DLinear

# python -u run.py \
#   --task_name sorting \
#   --des 'Exp' \
#   --is_training 1 \
#   --root_path ./dataset/EBDSC-2nd/ \
#   --model_id EBDSC_2nd \
#   --data EBDSC_2nd \
#   --data_regen_epoch 5 \
#   --seq_len 1024 \
#   --enc_in 5 \
#   --c_out 12 \
#   --model $model_name \
#   --e_layers 16 \
#   --batch_size 512 \
#   --d_model 128 \
#   --d_ff 512 \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 400 \
#   --patience 3


python -u run.py \
  --task_name sorting \
  --des 'vpos' \
  --is_training 1 \
  --root_path ./dataset/EBDSC-2nd/ \
  --model_id EBDSC_2nd \
  --data EBDSC_2nd \
  --data_regen_epoch 10 \
  --embed vpos \
  --seq_len 1024 \
  --enc_in 5 \
  --c_out 12 \
  --model $model_name \
  --e_layers 8 \
  --batch_size 64 \
  --d_model 128 \
  --d_ff 512 \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 400 \
  --patience 3
