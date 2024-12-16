export CUDA_VISIBLE_DEVICES=0

model_name=DLinear


python -u run.py \
  --checkpoint ./results/ \
  --task_name sorting \
  --des 'value' \
  --is_training 1 \
  --root_path ./dataset/EBDSC-2nd/ \
  --model_id EBDSC_2nd \
  --data EBDSC_2nd \
  --embed value \
  --wve_d_model 128 \
  --d_model 128 \
  --enc_in 5 \
  --seq_len 1024 \
  --c_out 12 \
  --model $model_name \
  --dlinear_individual \
  --e_layers 2 \
  --batch_size 16 \
  --d_ff 512 \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 400 \
  --patience 3
