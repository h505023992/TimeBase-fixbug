export CUDA_VISIBLE_DEVICES=0  # 

#cd ..

for model_name in FEDformer
do
for seq_len in 96; do
for pred_len in 96 
do

  
  label_len=$(( $seq_len / 2 ))
  python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh1.csv \
      --task_id Etth1_${seq_len}_$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len ${seq_len} \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --train_epochs 1 \
      --itr 1  >logs/$model_name'_Etth1_'${seq_len}'_'$pred_len.log
done
done
done