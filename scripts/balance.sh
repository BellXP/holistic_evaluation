model_name=$1
device=${2:-0}
batch_size=${3:-64}


python eval.py --model_name G2PT-3B --device 5 --batch_size 16 --eval_vqa --dataset_name OCR0 && python eval.py --model_name G2PT-3B --device 5 --batch_size 16 --eval_vqa --dataset_name OCR3 && python eval.py --model_name G2PT-3B --device 5 --batch_size 16 --eval_vqa --dataset_name OCR4

















# MRR

