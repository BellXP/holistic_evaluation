NUM_GPU=4
export CUDA_VISIBLE_DEVICES=4,5,6,7

######################
# ImageBind_7B-p7-f3 #
######################
python -m torch.distributed.launch \
--nproc_per_node=$NUM_GPU \
eval_knn.py \
--model_name ImageBind_7B-p7-f3 \
--vision_layer_index 3 \
--gather-on-cpu \
--batch-size 512

python -m torch.distributed.launch \
--nproc_per_node=$NUM_GPU \
eval_knn.py \
--model_name ImageBind_7B-p7-f3 \
--vision_layer_index 2 \
--gather-on-cpu \
--batch-size 512

python -m torch.distributed.launch \
--nproc_per_node=$NUM_GPU \
eval_knn.py \
--model_name ImageBind_7B-p7-f3 \
--vision_layer_index 1 \
--gather-on-cpu \
--batch-size 512

python -m torch.distributed.launch \
--nproc_per_node=$NUM_GPU \
eval_knn.py \
--model_name ImageBind_7B-p7-f3 \
--vision_layer_index 0 \
--gather-on-cpu \
--batch-size 512
