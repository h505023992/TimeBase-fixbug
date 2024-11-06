model_name=GPT2
train_epochs=30
llama_layers=12
learning_rate=0.001
master_port=12345
num_process=1
batch_size=32
d_model=32
d_ff=128
#CUDA_VISIBLE_DEVICES=1 
#export CUDA_VISIBLE_DEVICES=7  # 指定第7号GPU

for seq_len in 96 192 336 720; do
comment='TimeLLM-ETTh1'
for pred_len in 96 192 336 720; do
accelerate launch --gpu_ids 6  --num_processes=1   --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_${seq_len}_${pred_len} \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len ${seq_len} \
  --label_len 48 \
  --pred_len $pred_len \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment > ./logs/ETTh1_${seq_len}_${pred_len}.log
done


comment='TimeLLM-ETTh2'
for pred_len in 96 192 336 720; do
accelerate launch --gpu_ids 6  --num_processes=1   --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_${seq_len}_${pred_len} \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len ${seq_len} \
  --label_len 48 \
  --pred_len $pred_len \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment > ./logs/ETTh2_${seq_len}_${pred_len}.log
done
##上次从ettm开始断的


comment='TimeLLM-ETTm1'
for pred_len in 96 192 336 720; do
accelerate launch --gpu_ids 6  --num_processes=1   --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_${seq_len}_${pred_len} \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len ${seq_len} \
  --label_len 48 \
  --pred_len $pred_len \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment > ./logs/ETTm1_${seq_len}_${pred_len}.log
done

comment='TimeLLM-ETTm2'
for pred_len in 96 192 336 720; do
accelerate launch --gpu_ids 6  --num_processes=1   --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_${seq_len}_${pred_len} \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len ${seq_len} \
  --label_len 48 \
  --pred_len $pred_len \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment > ./logs/ETTm2_${seq_len}_${pred_len}.log
done
done