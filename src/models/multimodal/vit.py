import torch
import torch.nn as nn
from transformers import ViTModel


class ViT(nn.Module):
    def __init__(self, config = None):
        super(ViT, self).__init__()
        if(config == None):
            config = {
                "name":"vit",
                "ckpt":"/workspace/lpf/CLM-insights/ckpts/image_ckpts/vit-base-patch16-224"
            }
        self.main_model = ViTModel.from_pretrained(config['ckpt'])

    def forward(self, x):
        output = self.main_model(x)
        return output.last_hidden_state

