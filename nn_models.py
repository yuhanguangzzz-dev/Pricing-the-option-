import torch
import torch.nn as nn

class PyramidNN(nn.Module):
    """
    遵循 Masters (1993) 几何金字塔规则构建网络 (Section 3.3) [cite: 243]
    """
    def __init__(self, arch_type="NN3", input_dim=2):
        super().__init__()
        # 架构定义 [cite: 245-249]
        configs = {
            "NN1": [32],
            "NN2": [32, 16],
            "NN3": [32, 16, 8],
            "NN4": [32, 16, 8, 4],
            "NN5": [32, 16, 8, 4, 2]
        }
        units = configs[arch_type]
        layers = []
        curr_dim = input_dim
        for u in units:
            layers.append(nn.Linear(curr_dim, u))
            layers.append(nn.Sigmoid()) # 选用 Sigmoid 激活函数 [cite: 267]
            curr_dim = u
        layers.append(nn.Linear(curr_dim, 1)) # 输出修正项 f(.) [cite: 222]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)