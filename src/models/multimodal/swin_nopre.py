import torch
import torch.nn as nn
from transformers import SwinModel as swin


class SwinModel(nn.Module):
    def __init__(self, config = None):
        super(SwinModel, self).__init__()
        if(config == None):
            config = {
                "name":"swin_nopre",
                "ckpt":"/workspace/lpf/CLM-insights/ckpts/image_ckpts/swin-tiny"
            }
        self.main_model = swin.from_pretrained(config['ckpt'])

    def forward(self, x):
        output = self.main_model(x)
        return output.last_hidden_state

