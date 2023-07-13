model_name=$1
device=${2:-0}
batch_size=${3:-64}

# OCR
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr # 27867
# Caption
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_caption --dataset_name NoCaps # 4500
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_caption --dataset_name Flickr --sample_num 1000 # 1000
# KIE
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_kie --dataset_name SROIE # 347
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_kie --dataset_name FUNSD # 50
# VQA
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name TextVQA # 5000
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name DocVQA # 5349
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name STVQA --sample_num 4000 # 4000
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name OKVQA # 5046
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name ScienceQA # 2017
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name GQA # 12578
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name VizWiz # 1131
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name IconQA # 6316
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name VSR # 10972
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name OCRVQA # 100037
# MRR
python eval.py --model_name $model_name --device $device --batch_size $batch_size --eval_mrr --dataset_name Visdial # 20640
