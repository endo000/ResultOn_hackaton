
import boto3

import os
from fastapi import HTTPException
import os

import boto3

from botocore.client import Config


def s3_upload_file(file):
    # variable to save image name
    filename = 0
    try:
        #open image file
        contents = file.file.read()
        file.file.seek(0)

        # startup hte boto3 client
        s3 = boto3.client('s3',
                    endpoint_url=os.environ['MINIO_URI'],
                    aws_access_key_id=os.environ['MINIO_ACCESS_KEY'],
                    aws_secret_access_key=os.environ['MINIO_SECRET_ACCESS_KEY'],
                    config=Config(signature_version='s3v4'),
                    region_name='us-east-1')
        
        # upoad image
        s3.upload_fileobj(file.file, 'images', file.filename)

        # save image name
        filename = file.filename

    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        file.file.close()

    return filename