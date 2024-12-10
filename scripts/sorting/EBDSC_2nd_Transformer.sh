export CUDA_VISIBLE_DEVICES=4

model_name=Transformer


python -u run.py \
  --task_name sorting \
  --is_training 1 \
  --root_path ./dataset/EBDSC-2nd/ \
  --model_id EBDSC_2nd \
  --data EBDSC_2nd \
  --data_regen_epoch 10 \
  --des 'vpos' \
  --embed vpos \
  --seq_len 1024 \
  --enc_in 5 \
  --c_out 12 \
  --model $model_name \
  --e_layers 4 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 400 \
  --patience 2 \
  --use_amp