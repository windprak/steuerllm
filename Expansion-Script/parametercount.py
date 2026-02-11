import torch
from transformers import AutoModelForCausalLM

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return total_params, trainable_params, non_trainable_params

def format_params(n):
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.2f}K"
    else:
        return str(n)

model_name = "/hnvme/workspace/unrz103h-hnvme/anvme/base_models/mistralsmall"  # change to any model you want

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cpu"  # avoid GPU memory usage
)

total, trainable, frozen = count_parameters(model)

print(f"Model: {model_name}")
print(f"Total parameters:        {format_params(total)} ({total})")
print(f"Trainable parameters:    {format_params(trainable)} ({trainable})")
print(f"Non-trainable parameters:{format_params(frozen)} ({frozen})")
