# ResultOn_hackaton

## Project arhitecture
![Project architecture](/images/Hackaton_app_architecture.drawio.png)

## Backend

For local development and further deployment there is a [docker-compose.yml](backend/docker-compose.yml).

It reads parameters from an [.env](backend/.env) file. As it includes secrets, it is only an example file and parameters should be added manually before running services.

Currently next services are configured:

- **NGINX**
- **PostreSQL**
- **MinIO**
- **RabbitMQ**
- **MLFlow**

### NGINX

Dummy proxy for testing server accessibility.

### PostgreSQL

Stores MLflow backend data.

### MinIO

S3 server for development phase. Will be migrated to [Arnes Shramba](https://www.arnes.si/storitve/splet-posta-strezniki/arnes-shramba/).

Used for saving artifacts from MLFlow, as such model training data and model registry.

### RabbitMQ

Message broker, used for log sending from ML scripts.

### MLFlow

MLOps platform, used for saving trainings from classification model and model registry.

## Arnes HPC

Part of the project to run ML scripts. Uses Apptainer for building and running containers.

To build container run [build_apptainer_python.sh](hpc/build_apptainer_python.sh).

Bash scripts to run ML scripts are located [here](hpc/bin).

### [Alpaca](hpc/src/alpaca_llm/main.py)

### [Stable diffusion](hpc/src/stable_diffusion/main.py)

### [MobileNevV3](hpc/src/mobilenetv3/main.py)

Train and save MobileNetV3 fine tuned model

## Extra documentation

[Navodila za oddajo izdelkov iz 1. kroga HackathONa](docs/navodila_1_krog.md)
