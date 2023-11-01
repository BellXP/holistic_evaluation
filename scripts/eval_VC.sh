model_name=${1-"LLaVA"}
batch_size=${2-1}


######################
# Visual Commonsense #
######################

scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name ImageNetVC_color # 5570
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name ImageNetVC_shape # 4240
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name ImageNetVC_material # 4300
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name ImageNetVC_component # 11140
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name ImageNetVC_others # 15510