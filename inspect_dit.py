import torch
from models import DiT_models

model = DiT_models["DiT-XL/2"](
    input_size=32,
    num_classes=1000
)

print(model)