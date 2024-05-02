FROM ghcr.io/mlflow/mlflow:v2.12.1
RUN apt update && apt install -y gcc libpq-dev
RUN pip install psycopg2 boto3
