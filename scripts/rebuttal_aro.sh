device=${1:-0}
batch_size=${2:-8}

dataset_name="VG_Relation,VG_Attribution,COCO_Order,Flickr30k_Order"
model_names=('BLIP2' 'InstructBLIP' 'LLaMA-Adapter-v2' 'LLaVA' 'MiniGPT-4' 'mPLUG-Owl' 'Otter' 'VPGTrans')
for model_name in ${model_names[@]}
do
    python eval_rebuttal.py --model_name $model_name --device $device --batch_size $batch_size --dataset_name $dataset_name --sample_num 50
done
