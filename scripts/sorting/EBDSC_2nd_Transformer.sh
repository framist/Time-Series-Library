export CUDA_VISIBLE_DEVICES=0

model_name=Transformer

python -u run.py \
  --task_name sorting \
  --is_training 1 \
  --root_path ./dataset/EBDSC-2nd/ \
  --model_id EBDSC_2nd \
  --data EBDSC_2nd \
  --data_regen_epoch 10 \
  --des 'wve_cat_as_c' \
  --embed wve cat_as_c \
  --wve_d_model 128 \
  --d_model 128 \
  --enc_in 640 \
  --seq_len 1024 \
  --c_out 12 \
  --model $model_name \
  --e_layers 4 \
  --batch_size 64 \
  --d_ff 256 \
  --n_heads 4 \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 400 \
  --patience 100 \
  --use_amp

python -u run.py \
  --task_name sorting \
  --is_training 1 \
  --root_path ./dataset/EBDSC-2nd/ \
  --model_id EBDSC_2nd \
  --data EBDSC_2nd \
  --data_regen_epoch 10 \
  --des 'value' \
  --embed value \
  --wve_d_model 128 \
  --enc_in 5 \
  --d_model 128 \
  --seq_len 1024 \
  --c_out 12 \
  --model $model_name \
  --e_layers 4 \
  --batch_size 64 \
  --d_ff 256 \
  --n_heads 4 \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 400 \
  --patience 100 \
  --use_amp