model_name=$1
device=${2:-0}
batch_size=${3:-64}

# OCR
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr
# Caption
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_caption --dataset_name NoCaps
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_caption --dataset_name Flickr
# KIE
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_kie --dataset_name SROIE
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_kie --dataset_name FUNSD
# VQA
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name TextVQA
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name DocVQA
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name STVQA
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name OKVQA
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name ScienceQA
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name GQA
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name VizWiz
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name IconQA
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name VSR
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name OCRVQA
# MRR
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_mrr --dataset_name Visdial
