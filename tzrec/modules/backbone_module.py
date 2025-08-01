import torch
import torch.nn as nn

class Add(nn.Module):
    def forward(self, *inputs):
        # 支持输入为 list/tuple
        out = inputs[0]
        for i in range(1, len(inputs)):
            out = out + inputs[i]
        return out
    

