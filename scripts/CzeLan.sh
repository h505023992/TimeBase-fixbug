if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=LightTimeBaseTST
root_path_name=./dataset/
data_path_name=CzeLan.csv
model_id_name=CzeLan
data_name=custom
seq_len=720
gpu=3
pred_len=96
if [ ! -d "./logs/${model_id_name}" ]; then
    mkdir ./logs/${model_id_name}
fi
dir=./logs/${model_id_name}
for b in 6 ; do
for ind in 0 ; do
for pred_len in 96 192 336 720; do
python -u run_longExp.py \
    --is_training 1 \
    --orthogonal_weight 0.04 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model $model_name \
    --data $data_name \
    --features M \
    --individual $ind \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --period_len 4 \
    --use_period_norm 0 \
    --enc_in 11 \
    --individual $ind\
    --train_epochs 30 \
    --patience 5 \
    --basis_num $b \
    --gpu $gpu \
    --itr 1 --batch_size 512 --learning_rate 5e-2 > "$dir/${model_id_name}_${seq_len}_${pred_len}_${b}_${ind}.log"
done
done
done