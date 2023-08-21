PARTITION=gvlab
JOB_NAME=eval_llama2
NOW="`date +%Y%m%d%H%M%S`"
log_dir='/mnt/petrelfs/xupeng/workplace/logs'
PYTHON=/mnt/cache/xupeng/miniconda3/envs/accessory/bin/python
NUM_GPU=2

exec_params=${@}

# quotatype: 1.spot - other partition; 2.reserved - selected partition
# --async -o $log_dir/${JOB_NAME}_${NOW}.log
srun --partition=${PARTITION} --mpi=pmi2 --gres=gpu:$NUM_GPU -n$NUM_GPU --ntasks-per-node=$NUM_GPU \
--cpus-per-task=10 --job-name=${JOB_NAME} --quotatype=spot --async -o $log_dir/${JOB_NAME}_${NOW}.log \
$PYTHON eval_mme.py $exec_params

sleep 2