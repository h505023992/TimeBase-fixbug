if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./logs/${model_id_name}" ]; then
    mkdir ./logs/${model_id_name}
fi
dir=./logs/${model_id_name}
model_name=LightTimeBaseTST
root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
seq_len=336
gpu=1
pred_len=96
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
    --period_len 4 \
    --enc_in 7 \
    --individual 1 \
    --train_epochs 30 \
    --patience 5 \
    --basis_num 20 \
    --gpu $gpu \
    --itr 1 \
    --batch_size 512 \
    --learning_rate 2e-2 > "$dir/${model_id_name}_${seq_len}_${pred_len}.log"
pred_len=192
python -u run_longExp.py \
    --is_training 1 \
    --orthogonal_weight 0.08 \
    --root_path "$root_path_name" \
    --data_path "$data_path_name" \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model "$model_name" \
    --data "$data_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --period_len 4 \
    --enc_in 7 \
    --train_epochs 30 \
    --patience 5 \
    --basis_num 32 \
    --gpu $gpu \
    --itr 1 \
    --batch_size 512 \
    --learning_rate 2e-2 > "$dir/${model_id_name}_${seq_len}_${pred_len}.log"
pred_len=336
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
    --period_len 4 \
    --enc_in 7 \
    --individual 1 \
    --train_epochs 30 \
    --patience 5 \
    --basis_num 20 \
    --gpu $gpu \
    --itr 1 \
    --batch_size 512 \
    --learning_rate 2e-2 > "$dir/${model_id_name}_${seq_len}_${pred_len}.log"
pred_len=720
python -u run_longExp.py \
    --is_training 1 \
    --orthogonal_weight 0.08 \
    --root_path "$root_path_name" \
    --data_path "$data_path_name" \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model "$model_name" \
    --data "$data_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --period_len 6 \
    --enc_in 7 \
    --train_epochs 30 \
    --patience 5 \
    --basis_num 48 \
    --gpu $gpu \
    --itr 1 \
    --batch_size 512 \
    --learning_rate 5e-2 > "$dir/${model_id_name}_${seq_len}_${pred_len}.log"