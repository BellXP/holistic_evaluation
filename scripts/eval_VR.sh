model_name=${1-"LLaVA"}
batch_size=${2-1}


####################
# Visual Reasoning #
####################

scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name DocVQA # 5349
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name TextVQA # 5000
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name STVQA # 26074
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name OCRVQA # 100037
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name OKVQA # 5046
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name GQA # 12578
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name IconQA # 6316
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name VSR # 10972
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name WHOOPS # 3362
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name ScienceQA # 2017
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name VizWiz # 1131