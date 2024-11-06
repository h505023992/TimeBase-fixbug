#!/bin/bash
SCRIPT_DIR="./scripts/all"

SCRIPTS=( "traffic.sh" "electricity.sh" "weather.sh" "etth.sh") #

for script in "${SCRIPTS[@]}"; do
    echo "正在运行: $script"
    bash "$SCRIPT_DIR/$script"
    

    if [ $? -ne 0 ]; then
        echo "脚本 $script 执行失败！停止运行。" >&2
        exit 1  
    fi

    echo "$script 运行完成。"
done

echo "所有脚本执行完成。"
