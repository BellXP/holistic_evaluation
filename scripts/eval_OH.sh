model_name=${1-"LLaVA"}
batch_size=${2-1}


########################
# Object Hallucination #
########################

scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name MSCOCO_pope_random # 2910
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name MSCOCO_pope_popular # 3000
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name MSCOCO_pope_adversarial # 3000
