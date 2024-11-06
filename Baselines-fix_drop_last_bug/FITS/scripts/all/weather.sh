export CUDA_VISIBLE_DEVICES=6

# add for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi


model_name=FITS
data=weather
bs=64
seed=0
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
  --enc_in 21 \
  --des 'Exp' \
  --train_mode 2 \
  --H_order 12 \
  --base_T 144 \
  --gpu 0 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 --seed $seed > ./logs/${data}_${seq_len}_${pred_len}.log
  done
  done 