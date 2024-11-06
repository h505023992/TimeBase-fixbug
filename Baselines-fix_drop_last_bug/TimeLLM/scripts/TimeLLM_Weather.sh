model_name=TimeLLM
train_epochs=10
learning_rate=0.001
llama_layers=12

master_port=29502
num_process=8
batch_size=256
d_model=16
d_ff=32

comment='TimeLLM-Weather'
for pred_len in 96 192 336 720; do
accelerate launch --gpu_ids 3  --num_processes=1 --main_process_port $master_port   --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_720_${pred_len} \
  --model $model_name \
  --data Weather \
  --features M \
  --seq_len 720 \
  --label_len 48 \
  --pred_len ${pred_len} \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --d_ff 32 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment > ./logs/weather_720_${pred_len}.log
done