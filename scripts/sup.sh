# BLIP2
model_name="BLIP2"
batch_size=1

# BLIVA
model_name="BLIVA"
batch_size=16

# Cheetah
model_name="Cheetah"
batch_size=16
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name VSR

# InstructBLIP
model_name="InstructBLIP"
batch_size=1

# InstructBLIP-T5
model_name="InstructBLIP-T5"
batch_size=1

# LLaMA-Adapter-v2
model_name="LLaMA-Adapter-v2"
batch_size=1

# LLaVA
model_name="LLaVA"
batch_size=16

# Lynx
model_name="Lynx"
batch_size=16
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name MSCOCO_MCI

# Otter-Image
model_name="Otter-Image"
batch_size=8
scripts/run_eval.sh --model_name $model_name --batch_size $batch_size --eval_vqa --dataset_name VCR1_MCI

# VPGTrans
model_name="VPGTrans"
batch_size=16
