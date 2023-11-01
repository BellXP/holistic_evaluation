model_name=${1-"LLaVA"}
batch_size=${2-1}


################################
# Visual Knowledge Acquisition #
################################

# OCR
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name IIIT5K # 3000
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name IC13 # 848
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name IC15 # 2077
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name Total-Text # 2215
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name CUTE80 # 288
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name SVT # 647
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name SVTP # 645
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name COCO-Text # 9842
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name WordArt # 1511
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name CTW # 1572
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name HOST # 2416
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name WOST # 2416
# KIE
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name SROIE # 1388
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name FUNSD # 588