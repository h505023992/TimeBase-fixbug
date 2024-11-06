model_name=TimeLLM
train_epochs=10
learning_rate=0.001
llama_layers=12

master_port=29501
num_process=8
batch_size=128
d_model=16
d_ff=32
comment='TimeLLM-Traffic'

for pred_len in 96 192 336 720; do
accelerate launch --gpu_ids 4 --main_process_port $master_port --num_processes=1   --mixed_precision bf16 run_main.py  \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_720_${pred_len} \
  --model $model_name \
  --data Traffic \
  --features M \
  --seq_len 720 \
  --label_len 48 \
  --pred_len ${pred_len} \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment > ./logs/traffic_720_${pred_len}.log
done