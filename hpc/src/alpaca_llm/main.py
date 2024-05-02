print('import')
from transformers import pipeline
from transformers.utils import logging
import torch

logging.set_verbosity_debug()

prompt = "Write an email about an alpaca that likes flan"
print('pipeline')

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))

while True:
    data = input("Please enter the message:\n")
    if 'Exit' == data:
        break

    exec(data)

model = pipeline(model="declare-lab/flan-alpaca-gpt4-xl")
print('model')
model(prompt, max_length=128, do_sample=True)

while True:
    data = input("Please enter the message:\n")
    if 'Exit' == data:
        break

    print(model(data, max_length=128))
