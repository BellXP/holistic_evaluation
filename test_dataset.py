from task_datasets import dataset_class_dict


task_dataset_names = {
    'Visual Perception': [
        'ImageNet', 'CIFAR10', 'OxfordIIITPet', 'Flowers102',
        'VCR1_OC', 'VCR1_MCI', 'MSCOCO_OC', 'MSCOCO_MCI'
    ],
    'Visual Knowledge Acquisition': [
        "IIIT5K", "IC13", "IC15", "Total-Text", "CUTE80",
        "SVT", "SVTP", "COCO-Text", "WordArt", "CTW",
        "HOST", "WOST", 'SROIE', 'FUNSD'
    ],
    'Visual Reasoning': [
        'DocVQA', 'TextVQA', 'STVQA', 'OCRVQA', 'OKVQA',
        'GQA', 'IconQA', 'VSR', 'WHOOPS', 'ScienceQA', 'VizWiz'
    ],
    'Visual Commonsense': [
        'ImageNetVC_others', 'ImageNetVC_color', 'ImageNetVC_shape',
        'ImageNetVC_material', 'ImageNetVC_component', 'VCR'
    ],
    'Object Hallucination': [
        'MSCOCO_pope_random', 'MSCOCO_pope_popular', 'MSCOCO_pope_adversarial'
    ]
}


def main():
    total_num = 0
    for capability_name in task_dataset_names:
        dataset_names = task_dataset_names[capability_name]
        capability_num = 0
        for dataset_name in dataset_names:
            dataset = dataset_class_dict[dataset_name]()
            print(f"{dataset_name}\t{len(dataset)}")
            capability_num += len(dataset)
        print(f"{capability_name}\t{capability_num}\n")
        total_num += capability_num
    print(f"In total: {total_num}")


if __name__ == "__main__":
    main()