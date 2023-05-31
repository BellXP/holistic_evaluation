NUM_GPU=6
export CUDA_VISIBLE_DEVICES=0,1,2,4,5,6

python -m torch.distributed.launch \
--nproc_per_node=$NUM_GPU \
eval_knn.py \
--model_name ImageBind_7B-p3-f0 \
--vision_layer_index 3 \
--gather-on-cpu \
--batch-size 256