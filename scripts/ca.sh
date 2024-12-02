if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=LightTimeBaseTST
root_path_name=./dataset/
data_path_name=ca
model_id_name=ca
data_name=ca
seq_len=720
gpu=4
if [ ! -d "./logs/${model_id_name}" ]; then
    mkdir ./logs/${model_id_name}
fi
dir=./logs/${model_id_name}
for basis_num in 6  ; do
for lr in 5e-3 1e-2 5e-2 1e-1; do
for pred_len in 96 192 336 720; do
python -u run_longExp.py \
    --is_training 1 \
    --orthogonal_weight 0.04 \
    --root_path "$root_path_name" \
    --data_path "$data_path_name" \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model "$model_name" \
    --data "$data_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --period_len 24 \
    --use_period_norm 0 \
    --enc_in 8600 \
    --train_epochs 30 \
    --patience 5 \
    --basis_num $basis_num \
    --gpu $gpu \
    --itr 1 \
    --years 2017_2018_2019_2020_2021 \
    --batch_size 128 \
    --learning_rate $lr > "$dir/${model_id_name}_${seq_len}_${pred_len}_${basis_num}_${lr}.log"
done
done
done