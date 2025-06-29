#!/bin/bash

# 定义要处理的文件夹编号列表
FOLDER_IDS=(1)

# 模型配置与权重
CONFIG="inference/customize/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
MODEL_URL="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"

# 根路径
BASE_PATH="datasets/person/customize/video"

for ID in "${FOLDER_IDS[@]}"; do
  # 格式化编号为五位数，例如 9 -> 00009
  FOLDER_ID=$(printf "%05d" "$ID")
  
  INPUT_DIR="${BASE_PATH}/${FOLDER_ID}/images"
  OUTPUT_DIR="${BASE_PATH}/${FOLDER_ID}"
  
  echo "Processing folder: $FOLDER_ID"
  python inference/customize/detectron2/projects/DensePose/apply_net.py show "$CONFIG" "$MODEL_URL" "$INPUT_DIR" "$OUTPUT_DIR" dp_segm -v
done
