dataset_name=$1
for file_name in `find ./tiny_answers/ -name "${dataset_name}.json"`
do
    IFS="/" read -a str_array <<< $file_name
    model_name=${str_array[2]}
    exp_dir=${str_array[3]}
    echo "The result of $model_name in $exp_dir"
    cat tiny_answers/$model_name/$exp_dir/result.json
    echo -e "\n\n"
done