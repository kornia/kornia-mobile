import kornia
import torch
op=torch.jit.script(kornia.rgb_to_grayscale)
torch.jit.save(op, 'firstop.pt') 
