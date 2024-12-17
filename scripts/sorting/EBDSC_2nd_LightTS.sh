export CUDA_VISIBLE_DEVICES=0

model_name=LightTS

# NOTE: arg e_layers is invalid for LightTS 

python -u run.py \
  --checkpoint ./results/ \
  --task_name sorting \
  --is_training 1 \
  --root_path ./dataset/EBDSC-2nd/ \
  --model_id EBDSC_2nd \
  --data EBDSC_2nd \
  --data_regen_epoch 2 \
  --des 'woEmb' \
  --embed value \
  --wve_d_model 128 \
  --d_model 128 \
  --enc_in 5 \
  --seq_len 1024 \
  --c_out 12 \
  --model $model_name \
  --extra_emb \
  --e_layers 8 \
  --batch_size 512 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 400 \
  --patience 100
