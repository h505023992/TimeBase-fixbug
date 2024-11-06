#export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
seq_len=720

for seq_len in 720 336 192 96; do
for pred_len in 96 192 336 720; do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id ECL_${seq_len}_${pred_len} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 64 \
    --learning_rate 0.0001 \
    --itr 1 >./logs/Electricity_${seq_len}_${pred_len}.log
done
done