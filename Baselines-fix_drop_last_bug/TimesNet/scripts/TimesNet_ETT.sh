export CUDA_VISIBLE_DEVICES=7

model_name=TimesNet

for data in ETTh1 ETTh2 ETTm1 ETTm2; do
for seq_len in 336 192 96; do
for pred_len in 96 192 336 720; do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/\
  --data_path ${data}.csv \
  --model_id ${data}_${seq_len}_96 \
  --model $model_name \
  --data ${data} \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 >./logs/${data}_${seq_len}_${pred_len}.log
done
done
done