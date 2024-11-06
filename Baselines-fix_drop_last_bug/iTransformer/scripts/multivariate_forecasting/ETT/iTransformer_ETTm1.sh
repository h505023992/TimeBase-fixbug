model_name=iTransformer

seq_len=720
for seq_len in 720 336 192 96; do
for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_${seq_len}_${pred_len} \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1  >./logs/ETTm1_${seq_len}_${pred_len}.log
done
done