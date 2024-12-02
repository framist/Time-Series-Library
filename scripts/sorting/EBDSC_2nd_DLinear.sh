export CUDA_VISIBLE_DEVICES=0

model_name=DLinear

python -u run.py \
  --task_name sorting \
  --is_training 1 \
  --root_path ./dataset/EBDSC-2nd/ \
  --model_id EBDSC_2nd \
  --data EBDSC_2nd \
  --seq_len 1024 \
  --enc_in 5 \
  --c_out 12 \
  --model $model_name \
  --e_layers 16 \
  --batch_size 512 \
  --d_model 128 \
  --d_ff 512 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 400 \
  --patience 100
