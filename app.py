from model import model,processor
from PIL import Image    
import requests
import torch

prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"

raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))