#!/bin/bash

# 定义要运行的脚本文件夹
SCRIPT_DIR="./scripts/multivariate_forecasting"

# 依次运行各个目录下的 .sh 文件
for folder in  "Traffic"  "Weather" "ECL" "ETT"  ; do
    echo "Running scripts in $folder..."
    for script in "$SCRIPT_DIR/$folder"/*.sh; do
        if [[ -f "$script" ]]; then
            echo "Executing $script..."
            bash "$script"
        else
            echo "No .sh files found in $folder."
        fi
    done
done

echo "All scripts executed."
