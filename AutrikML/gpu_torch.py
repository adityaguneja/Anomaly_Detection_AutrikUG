#Scipt to check if GPU is being detected by PYTorch.

import torch


print(torch.__version__)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")