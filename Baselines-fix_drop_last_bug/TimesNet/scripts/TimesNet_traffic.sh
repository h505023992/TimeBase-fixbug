export CUDA_VISIBLE_DEVICES=7

model_name=TimesNet
for seq_len in 336 192 96; do
for pred_len in 96 192 336 720; do #从336开始断开
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_${seq_len}_${pred_len} \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len ${pred_len} \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 64 \
  --d_ff 64 \
  --top_k 5 \
  --batch_size 8 \
  --des 'Exp' \
  --itr 1 >./logs/traffic_${seq_len}_${pred_len}.log
done
done