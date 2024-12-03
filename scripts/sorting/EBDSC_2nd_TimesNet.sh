export CUDA_VISIBLE_DEVICES=0

# for aug in jitter scaling permutation magwarp timewarp windowslice windowwarp rotation spawner dtwwarp shapedtwwarp wdba discdtw discsdtw
# do
# echo using augmentation: ${aug}

python -u run.py \
  --task_name sorting \
  --is_training 1 \
  --root_path ./dataset/EBDSC-2nd/ \
  --model_id EBDSC_2nd \
  --data EBDSC_2nd \
  --seq_len 1024 \
  --enc_in 5 \
  --c_out 12 \
  --model TimesNet \
  --e_layers 4 \
  --batch_size 128 \
  --d_model 64 \
  --d_ff 128 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.002 \
  --train_epochs 400 \
  --patience 400 \
  --use_amp
