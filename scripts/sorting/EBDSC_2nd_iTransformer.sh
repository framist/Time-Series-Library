export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer


# python -u run.py \
#   --task_name sorting \
#   --is_training 1 \
#   --root_path ./dataset/EBDSC-2nd/ \
#   --model_id EBDSC_2nd \
#   --data EBDSC_2nd \
#   --data_regen_epoch 2 \
#   --des 'value_sl128' \
#   --embed value \
#   --wve_d_model 128 \
#   --enc_in 5 \
#   --d_model 1024 \
#   --seq_len 128 \
#   --c_out 12 \
#   --model $model_name \
#   --e_layers 8 \
#   --batch_size 256 \
#   --d_ff 2048 \
#   --n_heads 32 \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 400 \
#   --patience 50 

# python -u run.py \
#   --task_name sorting \
#   --is_training 1 \
#   --root_path ./dataset/EBDSC-2nd/ \
#   --model_id EBDSC_2nd \
#   --data EBDSC_2nd \
#   --data_regen_epoch 2 \
#   --des 'value_sl256' \
#   --embed value \
#   --wve_d_model 128 \
#   --enc_in 5 \
#   --d_model 2048 \
#   --seq_len 256 \
#   --c_out 12 \
#   --model $model_name \
#   --e_layers 8 \
#   --batch_size 128 \
#   --d_ff 4096 \
#   --n_heads 64 \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 400 \
#   --patience 50 


python -u run.py \
  --task_name sorting \
  --is_training 1 \
  --root_path ./dataset/EBDSC-2nd/ \
  --model_id EBDSC_2nd \
  --data EBDSC_2nd \
  --data_regen_epoch 2 \
  --des 'value_sl1024_lr01' \
  --embed value \
  --wve_d_model 128 \
  --enc_in 5 \
  --d_model 2048 \
  --seq_len 1024 \
  --c_out 12 \
  --model $model_name \
  --e_layers 8 \
  --batch_size 512 \
  --d_ff 4096 \
  --n_heads 64 \
  --itr 1 \
  --learning_rate 0.01 \
  --train_epochs 400 \
  --patience 50 

python -u run.py \
  --task_name sorting \
  --is_training 1 \
  --root_path ./dataset/EBDSC-2nd/ \
  --model_id EBDSC_2nd \
  --data EBDSC_2nd \
  --data_regen_epoch 2 \
  --des 'value_sl1024_lr001' \
  --embed value \
  --wve_d_model 128 \
  --enc_in 5 \
  --d_model 2048 \
  --seq_len 1024 \
  --c_out 12 \
  --model $model_name \
  --e_layers 8 \
  --batch_size 512 \
  --d_ff 4096 \
  --n_heads 64 \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 400 \
  --patience 50 


python -u run.py \
  --task_name sorting \
  --is_training 1 \
  --root_path ./dataset/EBDSC-2nd/ \
  --model_id EBDSC_2nd \
  --data EBDSC_2nd \
  --data_regen_epoch 2 \
  --des 'value_sl1024' \
  --embed value \
  --wve_d_model 128 \
  --enc_in 5 \
  --d_model 4096 \
  --seq_len 1024 \
  --c_out 12 \
  --model $model_name \
  --e_layers 8 \
  --batch_size 256 \
  --d_ff 8192 \
  --n_heads 128 \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 400 \
  --patience 50 

python -u run.py \
  --task_name sorting \
  --is_training 1 \
  --root_path ./dataset/EBDSC-2nd/ \
  --model_id EBDSC_2nd \
  --data EBDSC_2nd \
  --data_regen_epoch 2 \
  --des 'value_sl1024' \
  --embed value \
  --wve_d_model 128 \
  --enc_in 5 \
  --d_model 8192 \
  --seq_len 1024 \
  --c_out 12 \
  --model $model_name \
  --e_layers 8 \
  --batch_size 256 \
  --d_ff 8192 \
  --n_heads 128 \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 400 \
  --patience 50 

# python -u run.py \
#   --task_name sorting \
#   --is_training 1 \
#   --root_path ./dataset/EBDSC-2nd/ \
#   --model_id EBDSC_2nd \
#   --data EBDSC_2nd \
#   --data_regen_epoch 2 \
#   --des 'value_sl512' \
#   --embed value \
#   --wve_d_model 128 \
#   --enc_in 5 \
#   --d_model 2048 \
#   --seq_len 512 \
#   --c_out 12 \
#   --model $model_name \
#   --e_layers 8 \
#   --batch_size 128 \
#   --d_ff 4096 \
#   --n_heads 64 \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 400 \
#   --patience 50 
  


# python -u run.py \
#   --task_name sorting \
#   --is_training 1 \
#   --root_path ./dataset/EBDSC-2nd/ \
#   --model_id EBDSC_2nd \
#   --data EBDSC_2nd \
#   --data_regen_epoch 2 \
#   --des 'value_sl256' \
#   --embed value \
#   --wve_d_model 128 \
#   --enc_in 5 \
#   --d_model 1024 \
#   --seq_len 256 \
#   --c_out 12 \
#   --model $model_name \
#   --e_layers 8 \
#   --batch_size 256 \
#   --d_ff 2048 \
#   --n_heads 32 \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 400 \
#   --patience 50 
