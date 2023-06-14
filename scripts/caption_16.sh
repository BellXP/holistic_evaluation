device=${1:-0}
batch_size=${2:-64}


python eval.py --model_name BLIP2 --device $device --batch_size $batch_size --eval_caption --dataset_name NoCaps
python eval.py --model_name BLIP2 --device $device --batch_size $batch_size --eval_caption --dataset_name Flickr --sample_num 1000

python eval.py --model_name InstructBLIP --device $device --batch_size $batch_size --eval_caption --dataset_name NoCaps
python eval.py --model_name InstructBLIP --device $device --batch_size $batch_size --eval_caption --dataset_name Flickr --sample_num 1000

python eval.py --model_name LLaMA-Adapter-v2 --device $device --batch_size $batch_size --eval_caption --dataset_name NoCaps
python eval.py --model_name LLaMA-Adapter-v2 --device $device --batch_size $batch_size --eval_caption --dataset_name Flickr --sample_num 1000

python eval.py --model_name LLaVA --device $device --batch_size $batch_size --eval_caption --dataset_name NoCaps
python eval.py --model_name LLaVA --device $device --batch_size $batch_size --eval_caption --dataset_name Flickr --sample_num 1000

python eval.py --model_name MiniGPT-4 --device $device --batch_size $batch_size --eval_caption --dataset_name NoCaps
python eval.py --model_name MiniGPT-4 --device $device --batch_size $batch_size --eval_caption --dataset_name Flickr --sample_num 1000

python eval.py --model_name mPLUG-Owl --device $device --batch_size $batch_size --eval_caption --dataset_name NoCaps
python eval.py --model_name mPLUG-Owl --device $device --batch_size $batch_size --eval_caption --dataset_name Flickr --sample_num 1000

python eval.py --model_name Otter --device $device --batch_size $batch_size --eval_caption --dataset_name NoCaps
python eval.py --model_name Otter --device $device --batch_size $batch_size --eval_caption --dataset_name Flickr --sample_num 1000

python eval.py --model_name VPGTrans --device $device --batch_size $batch_size --eval_caption --dataset_name NoCaps
python eval.py --model_name VPGTrans --device $device --batch_size $batch_size --eval_caption --dataset_name Flickr --sample_num 1000

