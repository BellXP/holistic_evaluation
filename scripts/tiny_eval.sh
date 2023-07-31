model_name=$1
device=${2:-0}
batch_size=${3:-64}

# Caption
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --dataset_name NoCaps,Flickr,MSCOCO_caption_karpathy,WHOOPSCaption
# KIE
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --dataset_name SROIE,FUNSD,POIE
# MCI, OC
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --dataset_name MSCOCO_MCI,VCR1_MCI,MSCOCO_OC,VCR1_OC
# Object
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --dataset_name MSCOCO_pope_adversarial,MSCOCO_pope_popular,MSCOCO_pope_random,ImageNetVC_color,ImageNetVC_component,ImageNetVC_material,ImageNetVC_others,ImageNetVC_shape
# VQA
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --dataset_name AOKVQAClose,AOKVQAOpen,DocVQA,GQA,OCRVQA,OKVQA,STVQA,TextVQA,VQAv2,VizWiz,WHOOPSVQA,WHOOPSWeird
# VQA (choices)
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --dataset_name HatefulMemes,IconQA,VSR,ScienceQAIMG
# Embod
# python eval_tiny.py --model_name $model_name --device $device --batch_size 1 --dataset_name MetaWorld,FrankaKitchen,Minecraft,VirtualHome,MinecraftPolicy
# CLS
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --dataset_name CIFAR10,CIFAR100,Flowers102,ImageNet,OxfordIIITPet
# OCR
python eval_tiny.py --model_name $model_name --device $device --batch_size $batch_size --dataset_name COCO-Text,CTW,CUTE80,HOST,IC13,IC15,IIIT5K,SVTP,SVT,Total-Text,WOST,WordArt