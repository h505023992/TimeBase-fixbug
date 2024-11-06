export CUDA_VISIBLE_DEVICES=6
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# if [ ! -d "./logs/FITS_fix/etth2_abl" ]; then
#     mkdir ./logs/FITS_fix/etth2_abl
# fi
# seq_len=700
model_name=FITS
H_order=6
m=2
seed=0
bs=64 #256 #32 64 # 128 256
for data in ETTh2 ETTh1 ETTm2 ETTm1; do
for seq_len in 336 192 96 720; do
for pred_len in 96 192 336 720; do
python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ${data}.csv \
  --model_id ${data}_$seq_len'_'$pred_len \
  --model $model_name \
  --data ${data} \
  --features M \
  --seq_len $seq_len \
  --pred_len ${pred_len} \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 5 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 > ./logs/${data}_${seq_len}_${pred_len}.log #| tee logs/FITS_fix/etth2_abl/$m'_'$model_name'_'Etth2_$seq_len'_'96'_H'$H_order'_bs'$bs'_s'$seed.log
done
done
done