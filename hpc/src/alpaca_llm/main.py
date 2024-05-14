import json

import fire
from gpt4all import GPT4All
from rabbitmq import RabbitMQ
from transformers.utils import logging


class AlpacaLLM:
    def __init__(
        self,
        model_id="declare-lab/flan-alpaca-gpt4-xl",
        max_length=128,
        do_sample=True,
    ):
        self.rabbit = RabbitMQ(routing_key="alpaca_llm")

        self.rabbit.publish(
            "Initialize pipeline",
            path="startup",
        )

        # self.model = pipeline(model=model_id)
        self.model = GPT4All("gpt4all-falcon-newbpe-q4_0.gguf")

        self.rabbit.publish(
            "Pipeline initialized",
            path="startup",
        )

        self.max_length = max_length
        self.do_sample = do_sample

    def run(self, prompt, additional_path=""):
        if additional_path and additional_path[-1] != ".":
            additional_path = f"{additional_path}."

        self.rabbit.publish(
            prompt,
            path=f"{additional_path}request",
        )
        response = self.model.generate(prompt=prompt)

        self.rabbit.publish(
            response,
            path=f"{additional_path}response",
        )

        self.rabbit.publish(
            "finish",
            path=f"{additional_path}finish",
        )


def categories(categories: list[str] = []):
    llm = AlpacaLLM()

    with llm.model.chat_session():
        prompt = "Here’s a formula for a Stable Diffusion image prompt: An image of [adjective] [subject] [doing action], [creative lighting style], detailed, realistic, trending on artstation, in style of [famous artist 1], [famous artist 2], [famous artist 3]."
        llm.run(prompt, additional_path="promptsetup")

        for category in categories:
            prompt = f"Write 5 Stable Diffusion prompts using the above formula with the subject being {category}"
            llm.run(prompt, additional_path=category)


def llm(prompt: str = "I am a software engineer"):
    llm = AlpacaLLM()
    llm.run(prompt)


if __name__ == "__main__":
    logging.set_verbosity_debug()
    fire.Fire(
        {
            "categories": categories,
            "llm": llm,
        }
    )
