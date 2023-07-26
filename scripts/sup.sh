device=${1:-0}
batch_size=${2:-8}

python eval_tiny.py --model_name InstructBLIP --device $device --batch_size $batch_size --dataset_name ScienceQAIMG
