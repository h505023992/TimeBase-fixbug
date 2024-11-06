export CUDA_VISIBLE_DEVICES=7
# add for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
model_name=FITS
data=traffic
H_order=8
m=2
seed=0
bs=64
for seq_len in 336 192 96 720; do
for pred_len in 96 192 336 720; do
python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ${data}.csv \
  --model_id ${data}_${seq_len}_${pred_len} \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 862 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 5 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 > ./logs/${data}_${seq_len}_${pred_len}.log
done
done