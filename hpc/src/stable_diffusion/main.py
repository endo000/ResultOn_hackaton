import json
import uuid

import fire
import torch
from accelerate import PartialState
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import logging


class SDPipeline:
    def __init__(self, batch_size=5):
        self.distributed_state = PartialState()

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        )
        self.pipe.to(self.distributed_state.device)
        self.pipe.unet = torch.compile(
            self.pipe.unet, mode="reduce-overhead", fullgraph=True
        )

        self.batch_size = batch_size

    def __call__(self, prompts, images_length):
        output_images = []
        with self.distributed_state.split_between_processes([prompts]) as prompt:
            print(prompt)
            print(f"Category: {prompt['category']}, Prompt: {prompt['prompt']}")

            for _ in range(images_length):
                images = self.pipe(prompt=prompt).images
                output_images.extend(images)

        return output_images


def read_input(input_file):
    with open(input_file, "r") as f:
        return json.load(f)


def main(input_file="input.json", batch_size=5):
    input_data = read_input(input_file)
    prompts = input_data["prompts"]
    images_length = input_data["images_length"]

    sdpipe = SDPipeline()
    images = sdpipe(prompts, images_length)
    for image in images:
        image.save(f"/exec/images/image_{uuid.uuid4()}.png")


if __name__ == "__main__":
    logging.set_verbosity_debug()
    fire.Fire(main)
