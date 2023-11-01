import pdb
import glob
import pickle


dataset_names = glob.glob('tiny_lvlm_datasets/*/dataset.pkl')
for dataset_name in dataset_names:
    dataset = pickle.load(open(dataset_name, 'rb'))
    if type(dataset[0]['gt_answers']) is list:
        print(dataset_name)
        pdb.set_trace()