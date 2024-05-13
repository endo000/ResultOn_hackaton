#!/bin/bash

cd "$(dirname "$0")/.." || exit 1

srun \
	--job-name alpaca_llm \
	--reservation hackathon \
	--partition gpu \
	--gpus 1 \
	--cpus-per-gpu 4 \
	--mem 20GB \
	apptainer exec --nv \
	--bind src/alpaca_llm:/exec \
	python.sif \
	python3.11 /exec/main.py
