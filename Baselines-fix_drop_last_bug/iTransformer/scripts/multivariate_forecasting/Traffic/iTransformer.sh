# export CUDA_VISIBLE_DEVICES=2

model_name=iTransformer
seq_len=720
for seq_len in 720 336 192 96; do
for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_${seq_len}_${pred_len} \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 256\
  --d_ff 256 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --itr 1 >./logs/Traffic_${seq_len}_${pred_len}.log
done
done