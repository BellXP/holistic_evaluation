device=${1:-0}
batch_size=${2:-8}

dataset_name="ImageNetC,ImageNetC_blur,ImageNetC_digital,ImageNetC_noise,ImageNetC_weather,ImageNetC_extra"
model_names=('BLIP2' 'InstructBLIP' 'LLaMA-Adapter-v2' 'LLaVA' 'MiniGPT-4' 'mPLUG-Owl' 'Otter' 'PandaGPT' 'VPGTrans')
for model_name in ${model_names[@]}
do
    python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --dataset_name $dataset_name
done

PYTHON=/nvme/share/leimeng/.conda/envs/pt201_leim/bin/python
$PYTHON eval_tiny.py --model_name OFv2_4BI --device $device --batch_size $batch_size --dataset_name $dataset_name