model_name="LLaMA2-Accessory"
device=0
batch_size=32

# OCR
scripts/s1_eval.sh --model_name $model_name --device $device --batch_size $batch_size --eval_ocr # 27867
# Caption
scripts/s1_eval.sh --model_name $model_name --device $device --batch_size $batch_size --eval_caption --dataset_name NoCaps # 4500
scripts/s1_eval.sh --model_name $model_name --device $device --batch_size $batch_size --eval_caption --dataset_name Flickr --sample_num 1000 # 1000
# KIE
scripts/s1_eval.sh --model_name $model_name --device $device --batch_size $batch_size --eval_kie --dataset_name SROIE # 347
scripts/s1_eval.sh --model_name $model_name --device $device --batch_size $batch_size --eval_kie --dataset_name FUNSD # 50
# VQA
# scripts/s1_eval.sh --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name TextVQA # 5000
# scripts/s1_eval.sh --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name DocVQA # 5349
# scripts/s1_eval.sh --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name STVQA --sample_num 4000 # 4000
# scripts/s1_eval.sh --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name OKVQA # 5046
# scripts/s1_eval.sh --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name ScienceQA # 2017
# scripts/s1_eval.sh --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name GQA # 12578
# scripts/s1_eval.sh --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name VizWiz # 1131
# scripts/s1_eval.sh --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name IconQA # 6316
# scripts/s1_eval.sh --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name VSR # 10972
# scripts/s1_eval.sh --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name OCRVQA # 100037
# MRR
# scripts/s1_eval.sh --model_name $model_name --device $device --batch_size $batch_size --eval_mrr --dataset_name Visdial # 20640
