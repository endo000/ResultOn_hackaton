import io
import json
import os
import uuid

import boto3
import fire
import torch
from accelerate import PartialState
from botocore.client import Config
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import logging


class SDPipeline:
    def __init__(self, batch_size=5):
        self.distributed_state = PartialState()

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
        )
        self.pipe.to(self.distributed_state.device)
        self.pipe.unet = torch.compile(
            self.pipe.unet,
            mode="reduce-overhead",
            fullgraph=True,
        )

        self.batch_size = batch_size

    def __call__(self, category, prompts, images_length):
        output_images = []
        for prompt in prompts:
            print(prompt)
            print(f"Category: {category}, Prompt: {prompt}")

            for _ in range(images_length):
                images = self.pipe(prompt=prompt).images
                output_images.extend(images)

        return output_images


def read_input(input_file):
    with open(input_file, "r") as f:
        return json.load(f)


def save_pil_s3(s3, category, image):
    in_mem_file = io.BytesIO()
    image.save(in_mem_file, format=image.format)
    in_mem_file.seek(0)

    s3.upload_fileobj(
        in_mem_file,
        "images",
        f"test/{uuid.uuid4()}.{image.format}",
    )


def main(input_file="input.json", batch_size=5):
    input_data = read_input(input_file)
    categories = input_data["categories"]
    images_length = input_data["images_length"]

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_URI"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

    sdpipe = SDPipeline()
    for category in categories:
        prompts = category["prompts"]
        cat = category["category"]
        images = sdpipe(cat, prompts, images_length)

        for image in images:
            save_pil_s3(s3, cat, image)


if __name__ == "__main__":
    logging.set_verbosity_debug()
    fire.Fire(main)
