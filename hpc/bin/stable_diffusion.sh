#!/bin/bash

srun \
	--job-name stable_diffusion \
	--reservation hackathon \
	--partition gpu \
	--gpus 1 \
	--cpus-per-gpu 4 \
	--mem 20GB \
	apptainer exec --nv \
	--bind src/stable_diffusion:/exec \
	python.sif \
	python3.11 /exec/main.py
