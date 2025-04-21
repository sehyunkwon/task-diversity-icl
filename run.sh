CONFIG_DIR="./task_configs/num=3"

for json_file in $(ls "$CONFIG_DIR"/*.json | sort)
do
    echo "Processing $json_file"
    python -m src.train -json_file_path "$json_file" -wandb -entity your_entity_name -project your_project_name
    if [ $? -ne 0 ]; then
        echo "Processing $json_file failed."
        continue
    else
        echo "Processing $json_file completed successfully."
    fi
done
