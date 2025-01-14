#!/bin/bash

declare -A model_vsrest
model_vsrest=( ["ridge"]="1 2 3" ["cnn"]="1 2 3" ["esm1b"]="0 1 2 3" ["esm1v"]="0 1 2 3" )

target_folder="CombGym/FLIP/splits/SaCas9/splits"


input_files=($(ls $target_folder/*.csv | xargs -n 1 basename | sed 's/\.csv$//'))


total_files=${#input_files[@]}


progress=0


for file in "${input_files[@]}"; do
   
    progress=$((progress + 1))
    python -c "from tqdm import tqdm; tqdm(total=$total_files, initial=$progress-1, unit='file').update(1)"

    echo "Processing file: $file"
    
   
    vsrest=$(echo "$file" | grep -oP '\d(?=vsrest)')
    echo "Extracted vsrest: $vsrest"
    
    for embedding_model in "esm1b" "esm1v"; do
        embedding_command="python baselines/embeddings/embeddings.py $file $embedding_model $vsrest --make_fasta --bulk_compute --concat_tensors --truncate"
        echo "Running: $embedding_command"
        $embedding_command
    done
    
   
    for model in "${!model_vsrest[@]}"; do
        echo "Checking model: $model with supported vsrests: ${model_vsrest[$model]}"
        if [[ ${model_vsrest[$model]} =~ (^|[[:space:]])$vsrest($|[[:space:]]) ]]; then
           
            train_command="python baselines/train_all.py $file $model"
            echo "Running: $train_command"
            $train_command
        fi
    done
done




