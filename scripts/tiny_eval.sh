model_name=$1
device=${2:-0}
batch_size=${3:-64}

# Caption
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_caption --dataset_name NoCaps
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_caption --dataset_name Flickr
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_caption --dataset_name MSCOCO_caption_karpathy
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_caption --dataset_name WHOOPSCaption
# KIE
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_kie --dataset_name SROIE
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_kie --dataset_name FUNSD
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_kie --dataset_name POIE
# MCI, OC
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name MSCOCO_MCI
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name VCR1_MCI
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name MSCOCO_OC
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name VCR1_OC
# Object
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name MSCOCO_pope_adversarial
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name MSCOCO_pope_popular
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name MSCOCO_pope_random
# VQA
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name AOKVQAClose
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name AOKVQAOpen
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name DocVQA
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name GQA
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name HatefulMemes
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name IconQA
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name OCRVQA
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name OKVQA
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name STVQA
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name ScienceQAIMG
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name TextVQA
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name VQAv2
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name VSR
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name VizWiz
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name WHOOPSVQA
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name WHOOPSWeird
# MRR
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_mrr --dataset_name Visdial
# Embod
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_embod --dataset_name MetaWorld
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_embod --dataset_name FrankaKitchen
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_embod --dataset_name Minecraft
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_embod --dataset_name VirtualHome
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_embod --dataset_name MinecraftPolicy
# CLS
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_cls --dataset_name CIFAR10
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_cls --dataset_name CIFAR100
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_cls --dataset_name Flowers102
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_cls --dataset_name ImageNet
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_cls --dataset_name OxfordIIITPet
# OCR
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name COCO-Text
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name CTW
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name CUTE80
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name HOST
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name IC13
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name IC15
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name IIIT5K
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name SVTP
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name SVT
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name Total-Text
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name WOST
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name WordArt