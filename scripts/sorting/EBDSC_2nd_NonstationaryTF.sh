export CUDA_VISIBLE_DEVICES=0

model_name=Nonstationary_Transformer

# TODO Nonstationary_Transformer embed bug here
python -u run.py \
  --checkpoint ./results/ \
  --task_name sorting \
  --is_training 1 \
  --root_path ./dataset/EBDSC-2nd/ \
  --model_id EBDSC_2nd \
  --data EBDSC_2nd \
  --data_regen_epoch 2 \
  --des 'wve_cat_as_c' \
  --embed wve cat_as_c \
  --wve_d_model 128 \
  --d_model 128 \
  --enc_in 640 \
  --seq_len 1024 \
  --c_out 12 \
  --model $model_name \
  --e_layers 1 \
  --d_ff 256 \
  --n_heads 4 \
  --factor 3 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2 \
  --itr 1 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --train_epochs 400 \
  --patience 50 \
  --use_amp


# python -u run.py \
#   --checkpoint ./results/ \
#   --task_name sorting \
#   --is_training 1 \
#   --root_path ./dataset/EBDSC-2nd/ \
#   --model_id EBDSC_2nd \
#   --data EBDSC_2nd \
#   --data_regen_epoch 2 \
#   --des 'value' \
#   --embed value \
#   --wve_d_model 128 \
#   --enc_in 5 \
#   --d_model 128 \
#   --seq_len 1024 \
#   --c_out 12 \
#   --model $model_name \
#   --e_layers 4 \
#   --d_ff 256 \
#   --n_heads 4 \
#   --factor 3 \
#   --p_hidden_dims 256 256 \
#   --p_hidden_layers 2 \
#   --itr 1 \
#   --batch_size 64 \
#   --learning_rate 0.001 \
#   --train_epochs 400 \
#   --patience 50 \
#   --use_amp