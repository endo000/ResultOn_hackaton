#!/bin/bash

cd "$(dirname "$0")/.." || exit 1

srun \
	--job-name mobilenetv3 \
	--reservation hackathon \
	--partition gpu \
	--gpus 1 \
	--cpus-per-gpu 4 \
	--mem 20GB \
	apptainer exec --nv \
	--env-file src/mobilenetv3/.env \
	--bind src/mobilenetv3:/exec \
	--bind dataset:/dataset \
	python.sif \
	python3.11 /exec/main.py train \
	--model_name contracts \
	--dataset_path /dataset/contracts \
	--output_path /exec/models
