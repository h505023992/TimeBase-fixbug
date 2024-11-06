export CUDA_VISIBLE_DEVICES=0  # 

#cd ..

for model_name in FEDformer
do
for seq_len in 336 192 96; do
for pred_len in 96 192 336 720 
do
  label_len=$(( $seq_len / 2 ))
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path traffic.csv \
    --task_id traffic_${seq_len}_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len ${seq_len} \
    --label_len ${label_len} \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 30 >logs/$model_name'_traffic_'${seq_len}'_'$pred_len.log
  python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh1.csv \
      --task_id Etth1_${seq_len}_$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len ${seq_len} \
      --label_len ${label_len} \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1  >logs/$model_name'_Etth1_'${seq_len}'_'$pred_len.log

  python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path electricity.csv \
      --task_id electricity_${seq_len}_$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len ${seq_len} \
      --label_len ${label_len} \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --itr 1 >logs/$model_name'_electricity_'${seq_len}'_'$pred_len.log



  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path weather.csv \
    --task_id weather_${seq_len}_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len ${seq_len} \
    --label_len ${label_len} \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 30 >logs/$model_name'_weather_'${seq_len}'_'$pred_len.log


  
  python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh2.csv \
      --task_id ETTh2_${seq_len}_$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len ${seq_len} \
      --label_len ${label_len} \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1  >logs/$model_name'_Etth2_'${seq_len}'_'$pred_len.log
  
  python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm1.csv \
      --task_id ETTm1_${seq_len}_$pred_len \
      --model $model_name \
      --data ETTm1 \
      --features M \
      --seq_len ${seq_len} \
      --label_len ${label_len} \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1  >logs/$model_name'_Ettm1_'${seq_len}'_'$pred_len.log

  python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm2.csv \
      --task_id ETTm2_${seq_len}_$pred_len \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --seq_len ${seq_len} \
      --label_len ${label_len} \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1  >logs/$model_name'_Ettm2_'${seq_len}'_'$pred_len.log
done
done
done