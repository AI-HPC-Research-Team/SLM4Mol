import torch
import torch.nn as nn
from transformers import ResNetModel


class ResNet(nn.Module):
    def __init__(self, config = None):
        super(ResNet, self).__init__()
        if(config == None):
            config = {
                "name":"restnet",
                "ckpt":"/workspace/lpf/CLM-insights/ckpts/image_ckpts/resnet-50"
            }
        self.main_model = ResNetModel.from_pretrained(config['ckpt'])

    def forward(self, x):
        output = self.main_model(x)
        return output.last_hidden_state
    