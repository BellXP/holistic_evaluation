model_names=('BLIP2' 'InstructBLIP' 'LLaMA-Adapter-v2' 'LLaVA' 'MiniGPT-4' 'mPLUG-Owl' 'OFv2_4BI' 'Otter' 'Otter-Image' 'PandaGPT' 'VPGTrans')
for model_name in ${model_names[@]}
do
    python result_eval.py --model_name $model_name --dataset_name NoCaps,Flickr,MSCOCO_caption_karpathy,WHOOPSCaption,SROIE,FUNSD,POIE,MSCOCO_MCI,VCR1_MCI,MSCOCO_OC,VCR1_OC,MSCOCO_pope_adversarial,MSCOCO_pope_popular,MSCOCO_pope_random,ImageNetVC_color,ImageNetVC_component,ImageNetVC_material,ImageNetVC_others,ImageNetVC_shape,AOKVQAClose,AOKVQAOpen,DocVQA,GQA,OCRVQA,OKVQA,STVQA,TextVQA,VQAv2,VizWiz,WHOOPSVQA,WHOOPSWeird,HatefulMemes,IconQA,VSR,ScienceQAIMG,CIFAR10,CIFAR100,Flowers102,ImageNet,OxfordIIITPet,COCO-Text,CTW,CUTE80,HOST,IC13,IC15,IIIT5K,SVTP,SVT,Total-Text,WOST,WordArt
done

