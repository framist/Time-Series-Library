export CUDA_VISIBLE_DEVICES=0

# for aug in jitter scaling permutation magwarp timewarp windowslice windowwarp rotation spawner dtwwarp shapedtwwarp wdba discdtw discsdtw
# do
# echo using augmentation: ${aug}

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/EEG/ \
  --model_id EEGloaderMix \
  --model TimesNet \
  --data EEGloaderMix \
  --e_layers 10 \
  --batch_size 128 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.002 \
  --train_epochs 50 \
  --patience 10 \
  --use_amp \
  --augmentation_ratio 1 \
  --jitter \
  --scaling

# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/FaceDetection/ \
#   --model_id FaceDetection \
#   --model TimesNet \
#   --data UEA \
#   --e_layers 2 \
#   --batch_size 16 \
#   --d_model 64 \
#   --d_ff 256 \
#   --top_k 3 \
#   --num_kernels 4 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 30 \
#   --patience 10

# python run.py \
# --task_name classification \
# --is_training 1 \
# --root_path ./dataset/Handwriting/ \
# --model_id Handwriting \
# --model TimesNet \
# --data UEA \
# --e_layers 2 \
# --batch_size 16 \
# --d_model 32 \
# --d_ff 64 \
# --top_k 3 \
# --des 'Exp' \
# --itr 1 \
# --learning_rate 0.001 \
# --train_epochs 30 \
# --patience 10

# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/Heartbeat/ \
#   --model_id Heartbeat \
#   --model TimesNet \
#   --data UEA \
#   --e_layers 3 \
#   --batch_size 16 \
#   --d_model 16 \
#   --d_ff 32 \
#   --top_k 1 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 30 \
#   --patience 10

# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/JapaneseVowels/ \
#   --model_id JapaneseVowels \
#   --model TimesNet \
#   --data UEA \
#   --e_layers 2 \
#   --batch_size 16 \
#   --d_model 16 \
#   --d_ff 32 \
#   --top_k 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 60 \
#   --patience 10

# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/PEMS-SF/ \
#   --model_id PEMS-SF \
#   --model TimesNet \
#   --data UEA \
#   --e_layers 6 \
#   --batch_size 16 \
#   --d_model 64 \
#   --d_ff 64 \
#   --top_k 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 30 \
#   --patience 10

# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/SelfRegulationSCP1/ \
#   --model_id SelfRegulationSCP1 \
#   --model TimesNet \
#   --data UEA \
#   --e_layers 3 \
#   --batch_size 16 \
#   --d_model 16 \
#   --d_ff 32 \
#   --top_k 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 30 \
#   --patience 10

# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/SelfRegulationSCP2/ \
#   --model_id SelfRegulationSCP2 \
#   --model TimesNet \
#   --data UEA \
#   --e_layers 3 \
#   --batch_size 16 \
#   --d_model 32 \
#   --d_ff 32 \
#   --top_k 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 30 \
#   --patience 10

# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/SpokenArabicDigits/ \
#   --model_id SpokenArabicDigits \
#   --model TimesNet \
#   --data UEA \
#   --e_layers 2 \
#   --batch_size 16 \
#   --d_model 32 \
#   --d_ff 32 \
#   --top_k 2 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 30 \
#   --patience 10

# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/UWaveGestureLibrary/ \
#   --model_id UWaveGestureLibrary \
#   --model TimesNet \
#   --data UEA \
#   --e_layers 2 \
#   --batch_size 16 \
#   --d_model 32 \
#   --d_ff 64 \
#   --top_k 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 30 \
#   --patience 10
