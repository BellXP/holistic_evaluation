from task_datasets import dataset_class_dict


def main():
    dataset_names = list(dataset_class_dict.keys())
    for dataset_name in dataset_names:
        dataset = dataset_class_dict[dataset_name]()
        print(f"[{dataset_name}]({len(dataset)}): {dataset[0]}")


if __name__ == "__main__":
    main()