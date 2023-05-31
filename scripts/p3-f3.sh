python eval.py --device 1 --batch_size 8 --model_name ImageBind_7B-p3-f3 --eval_vqa --dataset_name TextVQA
python eval.py --device 1 --batch_size 8 --model_name ImageBind_7B-p3-f3 --eval_vqa --dataset_name DocVQA
python eval.py --device 1 --batch_size 8 --model_name ImageBind_7B-p3-f3 --eval_vqa --dataset_name STVQA --sample_num 4000

python eval.py --device 1 --batch_size 8 --model_name ImageBind_7B-p3-f3 --eval_vqa --dataset_name ScienceQA
python eval.py --device 1 --batch_size 8 --model_name ImageBind_7B-p3-f3 --eval_caption --dataset_name NoCaps
python eval.py --device 1 --batch_size 8 --model_name ImageBind_7B-p3-f3 --eval_caption --dataset_name Flickr --sample_num 1000
python eval.py --device 1 --batch_size 8 --model_name ImageBind_7B-p3-f3 --eval_kie --dataset_name SROIE