import torch.nn.functional as F
import torch
import torch.nn as nn
# 线性分类器模型 
class LinearClassifier(nn.Module):
    def __init__(self, input_dim=4096, output_dim=1):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim) # 线性分类器
    def forward(self, x):
        output = self.linear(x)
        output = torch.sigmoid(output)
        
        return output.squeeze()
