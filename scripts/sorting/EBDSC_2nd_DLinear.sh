export CUDA_VISIBLE_DEVICES=0

model_name=DLinear

python -u run.py \
  --checkpoint ./results/ \
  --task_name sorting \
  --is_training 1 \
  --root_path ./dataset/EBDSC-2nd/ \
  --model_id EBDSC_2nd \
  --data EBDSC_2nd \
  --data_regen_epoch 2 \
  --des 'indiv_extraWve_cat_as_c' \
  --embed wve cat_as_c \
  --wve_d_model 128 \
  --enc_in 640 \
  --d_model 128 \
  --seq_len 1024 \
  --c_out 12 \
  --model $model_name \
  --dlinear_individual \
  --extra_emb \
  --e_layers 8 \
  --batch_size 64 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 400 \
  --patience 100

python -u run.py \
  --checkpoint ./results/ \
  --task_name sorting \
  --is_training 1 \
  --root_path ./dataset/EBDSC-2nd/ \
  --model_id EBDSC_2nd \
  --data EBDSC_2nd \
  --data_regen_epoch 2 \
  --des 'indiv_extraV' \
  --embed value \
  --wve_d_model 128 \
  --d_model 128 \
  --enc_in 5 \
  --seq_len 1024 \
  --c_out 12 \
  --model $model_name \
  --dlinear_individual \
  --extra_emb \
  --e_layers 8 \
  --batch_size 64 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 400 \
  --patience 100
