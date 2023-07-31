device=${1:-0}
batch_size=${2:-8}

dataset_name="IconQA,ImageNetVC_shape,ImageNetVC_color,ImageNetVC_component,ImageNetVC_material,ImageNetVC_others"
model_names=('BLIP2' 'InstructBLIP' 'LLaMA-Adapter-v2' 'LLaVA' 'MiniGPT-4' 'mPLUG-Owl' 'Otter' 'VPGTrans')
for model_name in ${model_names[@]}
do
    python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --dataset_name $dataset_name
done
