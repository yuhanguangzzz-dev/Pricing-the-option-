import torch
import torch.nn as nn

import torch
import torch.nn as nn


class PyramidNN(nn.Module):
    """
    遵循 Masters (1993) 几何金字塔规则构建网络 (Section 3.3)
    横截面网络 (输入维度=2) ，面板数据特征网络 (输入维度=2+q)
    """

    def __init__(self, arch_type="NN3", num_features=0):
        super().__init__()
        # 如果 num_features = 0, 就是普通的 NN1-NN5
        # 如果 num_features = 8, 配合 arch_type="NN3", 就是论文中的 NN3F
        input_dim = 2 + num_features

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
            layers.append(nn.Sigmoid())  # 论文极度保守地选用了 Sigmoid
            curr_dim = u

        layers.append(nn.Linear(curr_dim, 1))  # 输出定价误差残差修正项 f(.)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)