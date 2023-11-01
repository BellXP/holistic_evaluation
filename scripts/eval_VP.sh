model_name=${1-"LLaVA"}
batch_size=${2-1}


#####################
# Visual Perception #
#####################

# classification
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name ImageNet # 50000
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name CIFAR10 # 10000
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name OxfordIIITPet # 3669
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name Flowers102 # 6149
# OC, MCI
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name VCR1_OC # 10000
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name VCR1_MCI # 10000
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name MSCOCO_OC # 10000
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name MSCOCO_MCI # 10000