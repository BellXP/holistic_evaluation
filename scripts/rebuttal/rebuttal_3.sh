model_name=$1
device=${2:-0}
batch_size=${3:-64}

python eval_rebuttal.py --model_name $model_name --device $device --batch_size $batch_size --dataset_name IsThereTest_No,IsThereTest_Yes