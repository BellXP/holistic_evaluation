model_name=$1
device=${2:-0}
batch_size=${3:-64}

PYTHON=/nvme/share/leimeng/.conda/envs/pt201_leim/bin/python

# Caption
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_caption --dataset_name NoCaps
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_caption --dataset_name Flickr
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_caption --dataset_name MSCOCO_caption_karpathy
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_caption --dataset_name WHOOPSCaption
# KIE
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_kie --dataset_name SROIE
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_kie --dataset_name FUNSD
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_kie --dataset_name POIE
# MCI, OC
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name MSCOCO_MCI
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name VCR1_MCI
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name MSCOCO_OC
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name VCR1_OC
# Object
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name MSCOCO_pope_adversarial
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name MSCOCO_pope_popular
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name MSCOCO_pope_random
# VQA
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name AOKVQAClose
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name AOKVQAOpen
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name DocVQA
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name GQA
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name HatefulMemes
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name IconQA
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name OCRVQA
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name OKVQA
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name STVQA
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name ScienceQAIMG
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name TextVQA
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name VQAv2
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name VSR
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name VizWiz
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name WHOOPSVQA
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_vqa --dataset_name WHOOPSWeird
# MRR
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_mrr --dataset_name Visdial
# Embod
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_embod --dataset_name MetaWorld
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_embod --dataset_name FrankaKitchen
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_embod --dataset_name Minecraft
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_embod --dataset_name VirtualHome
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_embod --dataset_name MinecraftPolicy
# CLS
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_cls --dataset_name CIFAR10
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_cls --dataset_name CIFAR100
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_cls --dataset_name Flowers102
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_cls --dataset_name ImageNet
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_cls --dataset_name OxfordIIITPet
# OCR
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name COCO-Text
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name CTW
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name CUTE80
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name HOST
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name IC13
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name IC15
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name IIIT5K
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name SVTP
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name SVT
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name Total-Text
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name WOST
$PYTHON eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --eval_ocr --dataset_name WordArt