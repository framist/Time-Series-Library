export CUDA_VISIBLE_DEVICES=0

# in the case of TimesNet, use_amp should be False

model_name=TimesNet

python -u run.py \
  --task_name sorting \
  --is_training 1 \
  --root_path ./dataset/EBDSC-2nd/ \
  --model_id EBDSC_2nd \
  --data EBDSC_2nd \
  --data_regen_epoch 2 \
  --des 'wve_cat_as_c' \
  --embed wve cat_as_c \
  --wve_d_model 128 \
  --enc_in 640 \
  --d_model 64 \
  --seq_len 1024 \
  --c_out 12 \
  --model $model_name \
  --e_layers 8 \
  --batch_size 256 \
  --d_ff 256 \
  --top_k 3 \
  --itr 1 \
  --learning_rate 0.002 \
  --train_epochs 400 \
  --patience 50 \
  --use_amp


python -u run.py \
  --task_name sorting \
  --is_training 1 \
  --root_path ./dataset/EBDSC-2nd/ \
  --model_id EBDSC_2nd \
  --data EBDSC_2nd \
  --data_regen_epoch 2 \
  --des 'value' \
  --embed value \
  --wve_d_model 128 \
  --enc_in 5 \
  --d_model 64 \
  --seq_len 1024 \
  --c_out 12 \
  --model $model_name \
  --e_layers 8 \
  --batch_size 256 \
  --d_ff 256 \
  --top_k 3 \
  --itr 1 \
  --learning_rate 0.002 \
  --train_epochs 400 \
  --patience 50 \
  --use_amp
