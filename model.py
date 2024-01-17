
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/bakLlava-v1-hf"


model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 

).to(0)


processor = AutoProcessor.from_pretrained(model_id)


