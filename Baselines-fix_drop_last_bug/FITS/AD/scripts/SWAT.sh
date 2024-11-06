# export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ft_SWAT_Full" ]; then
    mkdir ./logs/ft_SWAT_Full
fi

dsr=4
win_sizes=(400 200)
batch_sizes=(32 64 128 256)
lrs=(1e-4)
cutfreq=0

for win_size in ${win_sizes[@]}; do
    for batch_size in ${batch_sizes[@]}; do
        for lr in ${lrs[@]}; do
            python main.py --anormly_ratio 5 --num_epochs 50    --batch_size $batch_size  --mode train --dataset SWAT  --data_path dataset/SWAT_Full --input_c 55    --output_c 55  --win_size $win_size --DSR $dsr --lr $lr --cutfreq $cutfreq >logs/ft_SWAT_Full/training_bs${batch_size}_win${win_size}_ds${dsr}_lr${lr}_c${cutfreq}.log

            for ar in {1..10}; do
                python main.py --anormly_ratio $ar  --num_epochs 50       --batch_size $batch_size     --mode test    --dataset SWAT   --data_path dataset/SWAT_Full  --input_c 55    --output_c 55  --pretrained_model 20  --win_size $win_size --DSR $dsr --lr $lr --cutfreq $cutfreq >logs/ft_SWAT_Full/testing_ar${ar}_bs${batch_size}_win${win_size}_ds${dsr}_lr${lr}_c${cutfreq}.log
            done

        done
    done
done

# python main.py --anormly_ratio 5 --num_epochs 50    --batch_size 64  --mode train --dataset SWAT  --data_path dataset/SWAT --input_c 55    --output_c 55  --win_size 200 --DSR $dsr --lr 1e-4 >logs/ft_SWAT/training_bs64_win200_ds2.log

# python main.py --anormly_ratio 1  --num_epochs 10       --batch_size 64     --mode test    --dataset SWAT   --data_path dataset/SWAT  --input_c 55    --output_c 55  --pretrained_model 20  --win_size 200 --DSR $dsr --lr 5e-4 >logs/ft_SWAT/testing_ar1_bs64_win200_ds2.log
# python main.py --anormly_ratio 2  --num_epochs 10       --batch_size 64     --mode test    --dataset SWAT   --data_path dataset/SWAT  --input_c 55    --output_c 55  --pretrained_model 20  --win_size 200 --DSR $dsr --lr 5e-4 >logs/ft_SWAT/testing_ar2_bs64_win200_ds2.log
# python main.py --anormly_ratio 3  --num_epochs 10       --batch_size 64     --mode test    --dataset SWAT   --data_path dataset/SWAT  --input_c 55    --output_c 55  --pretrained_model 20  --win_size 200 --DSR $dsr --lr 5e-4 >logs/ft_SWAT/testing_ar3_bs64_win200_ds2.log
# python main.py --anormly_ratio 4  --num_epochs 10       --batch_size 64     --mode test    --dataset SWAT   --data_path dataset/SWAT  --input_c 55    --output_c 55  --pretrained_model 20  --win_size 200 --DSR $dsr --lr 5e-4 >logs/ft_SWAT/testing_ar4_bs64_win200_ds2.log
# python main.py --anormly_ratio 5  --num_epochs 10       --batch_size 64     --mode test    --dataset SWAT   --data_path dataset/SWAT  --input_c 55    --output_c 55  --pretrained_model 20  --win_size 200 --DSR $dsr --lr 5e-4 >logs/ft_SWAT/testing_ar5_bs64_win200_ds2.log
# python main.py --anormly_ratio 6  --num_epochs 10       --batch_size 64     --mode test    --dataset SWAT   --data_path dataset/SWAT  --input_c 55    --output_c 55  --pretrained_model 20  --win_size 200 --DSR $dsr --lr 5e-4 >logs/ft_SWAT/testing_ar6_bs64_win200_ds2.log
# python main.py --anormly_ratio 7  --num_epochs 10       --batch_size 64     --mode test    --dataset SWAT   --data_path dataset/SWAT  --input_c 55    --output_c 55  --pretrained_model 20  --win_size 200 --DSR $dsr --lr 5e-4 >logs/ft_SWAT/testing_ar7_bs64_win200_ds2.log
# python main.py --anormly_ratio 8  --num_epochs 10       --batch_size 64     --mode test    --dataset SWAT   --data_path dataset/SWAT  --input_c 55    --output_c 55  --pretrained_model 20  --win_size 200 --DSR $dsr --lr 5e-4 >logs/ft_SWAT/testing_ar8_bs64_win200_ds2.log
# python main.py --anormly_ratio 9  --num_epochs 10       --batch_size 64     --mode test    --dataset SWAT   --data_path dataset/SWAT  --input_c 55    --output_c 55  --pretrained_model 20  --win_size 200 --DSR $dsr --lr 5e-4 >logs/ft_SWAT/testing_ar9_bs64_win200_ds2.log
# python main.py --anormly_ratio 10 --num_epochs 10       --batch_size 64     --mode test    --dataset SWAT   --data_path dataset/SWAT  --input_c 55    --output_c 55  --pretrained_model 20  --win_size 200 --DSR $dsr --lr 5e-4 >logs/ft_SWAT/testing_ar10_bs64_win200_ds2.log
# python main.py --anormly_ratio 11 --num_epochs 10       --batch_size 64     --mode test    --dataset SWAT   --data_path dataset/SWAT  --input_c 55    --output_c 55  --pretrained_model 20  --win_size 200 --DSR $dsr --lr 5e-4 >logs/ft_SWAT/testing_ar11_bs64_win200_ds2.log
# python main.py --anormly_ratio 13 --num_epochs 10       --batch_size 64     --mode test    --dataset SWAT   --data_path dataset/SWAT  --input_c 55    --output_c 55  --pretrained_model 20  --win_size 200 --DSR $dsr --lr 5e-4 >logs/ft_SWAT/testing_ar13_bs64_win200_ds2.log

python main.py --anormly_ratio 5 --num_epochs 50    --batch_size 1  --mode train --dataset SWAT  --data_path dataset/SWAT_Full --input_c 55    --output_c 55  --win_size 200 --DSR 4 --lr 1 --cutfreq 0