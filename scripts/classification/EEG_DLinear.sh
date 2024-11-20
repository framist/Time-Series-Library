export CUDA_VISIBLE_DEVICES=0

model_name=DLinear

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/EEG/ \
  --model_id EEGloaderMix \
  --data EEGloaderMix \
  --model $model_name \
  --e_layers 3 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 20

# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/FaceDetection/ \
#   --model_id FaceDetection \
#   --model $model_name \
#   --data UEA \
#   --e_layers 3 \
#   --batch_size 16 \
#   --d_model 128 \
#   --d_ff 256 \
#   --top_k 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 100 \
#   --patience 10

# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/Handwriting/ \
#   --model_id Handwriting \
#   --model $model_name \
#   --data UEA \
#   --e_layers 3 \
#   --batch_size 16 \
#   --d_model 128 \
#   --d_ff 256 \
#   --top_k 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 100 \
#   --patience 10

# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/Heartbeat/ \
#   --model_id Heartbeat \
#   --model $model_name \
#   --data UEA \
#   --e_layers 3 \
#   --batch_size 16 \
#   --d_model 128 \
#   --d_ff 256 \
#   --top_k 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 100 \
#   --patience 10

# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/JapaneseVowels/ \
#   --model_id JapaneseVowels \
#   --model $model_name \
#   --data UEA \
#   --e_layers 3 \
#   --batch_size 16 \
#   --d_model 128 \
#   --d_ff 256 \
#   --top_k 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 100 \
#   --patience 10

# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/PEMS-SF/ \
#   --model_id PEMS-SF \
#   --model $model_name \
#   --data UEA \
#   --e_layers 3 \
#   --batch_size 16 \
#   --d_model 128 \
#   --d_ff 256 \
#   --top_k 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 100 \
#   --patience 10

# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/SelfRegulationSCP1/ \
#   --model_id SelfRegulationSCP1 \
#   --model $model_name \
#   --data UEA \
#   --e_layers 3 \
#   --batch_size 16 \
#   --d_model 128 \
#   --d_ff 256 \
#   --top_k 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 100 \
#   --patience 10

# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/SelfRegulationSCP2/ \
#   --model_id SelfRegulationSCP2 \
#   --model $model_name \
#   --data UEA \
#   --e_layers 3 \
#   --batch_size 16 \
#   --d_model 128 \
#   --d_ff 256 \
#   --top_k 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 100 \
#   --patience 10

# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/SpokenArabicDigits/ \
#   --model_id SpokenArabicDigits \
#   --model $model_name \
#   --data UEA \
#   --e_layers 3 \
#   --batch_size 16 \
#   --d_model 128 \
#   --d_ff 256 \
#   --top_k 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 100 \
#   --patience 10

# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/UWaveGestureLibrary/ \
#   --model_id UWaveGestureLibrary \
#   --model $model_name \
#   --data UEA \
#   --e_layers 3 \
#   --batch_size 16 \
#   --d_model 128 \
#   --d_ff 256 \
#   --top_k 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 100 \
#   --patience 10
