import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from torch.utils.data import random_split
class CustomConvAutoencoder(nn.Module):
    def __init__(self):
        super(CustomConvAutoencoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=(1,2), stride=1, padding=0), 
        )
        self.decoder = nn.Sequential(
            # 第一步: 上采样到 [batch_size, 32, 2, 2]
            nn.Upsample(scale_factor=2, mode='nearest'),
            # 适用卷积调整通道数，同时保持尺寸
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            # 第二步: 上采样到 [batch_size, 16, 4, 4]
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            # 第三步: 上采样到 [batch_size, 8, 8, 8]
            nn.Upsample(size=(8, 8), mode='nearest'),  
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            
            nn.Conv2d(4, 1, kernel_size=(1, 1), stride=1, padding=0),
        )

        # self.linear3 = nn.Linear(8 * 8, 7 * 9,bias=False)
        # self.linear1 = nn.Linear(9 * 12, 7 * 9,bias=False)
        # self.linear2 = nn.Linear(7 * 9, 7 * 9,bias=False)
        
        self.linear = nn.Linear(64, 63,bias=False)
       

    def forward(self, x):
        x = self.encoder(x)
        # x = self.decoder(x)
        # x = x.view(x.size(0), -1)
        # x = self.linear3(x)
        # x = x.view(-1, 1, 7, 9)
        return x

if __name__ == "__main__":
  model = CustomConvAutoencoder()

  # model.load_state_dict(torch.load('best_model.pth'))

  # 原始数据进入模型前需要先标准化
  means = {'CPU_percent': 8.74062442779541,
  'IO_read_standard': 6491806.0,
  'IO_write_standard': 6069067.5,
  'IO_read': 6491806.0,
  'IO_write': 3879825.5,
  'Memory_percent': 41.088340759277344,
  'Memory_used': 64620273664.0}
  stds = {'CPU_percent': 5.710987091064453,
  'IO_read_standard': 15870580.0,
  'IO_write_standard': 4721631.0,
  'IO_read': 15870579.0,
  'IO_write': 3051493.0,
  'Memory_percent': 0.38481253385543823,
  'Memory_used': 687050112.0}
  # data=(data-mean)/std

  #输入形状 (batch_size,7,9)
  input=torch.Tensor([[[ 0.5234, -0.6970, -0.3608, -0.0351, -0.0141, -0.1367, -0.5412,
            0.0717, -0.1682],
          [-0.1513, -0.2435,  0.5556, -0.2047,  0.4895, -0.2545, -0.3185,
            -0.2915, -0.2842],
          [-0.4025, -0.6780,  2.1815, -0.5836,  1.8659, -0.7740, -0.9894,
            -0.8962, -0.8433],
          [-0.1513, -0.2435,  0.5556, -0.2047,  0.4895, -0.2545, -0.3185,
            -0.2915, -0.2842],
          [-0.4047, -0.5858,  1.2240, -0.4226,  0.8204, -0.7415, -0.8366,
            -0.7956, -0.8764],
          [-0.0736, -0.1256, -0.0217, -0.1516, -0.0996, -0.1516, -0.1776,
            0.0303,  0.0303],
          [ 0.4242, -0.0615,  0.4260, -0.1198,  0.4673, -0.1930,  0.4885,
            -0.0554,  0.5659]],

          [[ 0.5234, -0.6970, -0.3608, -0.0351, -0.0141, -0.1367, -0.5412,
            0.0717, -0.1682],
          [-0.1513, -0.2435,  0.5556, -0.2047,  0.4895, -0.2545, -0.3185,
            -0.2915, -0.2842],
          [-0.4025, -0.6780,  2.1815, -0.5836,  1.8659, -0.7740, -0.9894,
            -0.8962, -0.8433],
          [-0.1513, -0.2435,  0.5556, -0.2047,  0.4895, -0.2545, -0.3185,
            -0.2915, -0.2842],
          [-0.4047, -0.5858,  1.2240, -0.4226,  0.8204, -0.7415, -0.8366,
            -0.7956, -0.8764],
          [-0.0736, -0.1256, -0.0217, -0.1516, -0.0996, -0.1516, -0.1776,
            0.0303,  0.0303],
          [ 0.4242, -0.0615,  0.4260, -0.1198,  0.4673, -0.1930,  0.4885,
            -0.0554,  0.5659]],

          [[-0.6970, -0.3608, -0.0351, -0.0141, -0.1367, -0.5412,  0.0717,
            -0.1682,  0.0787],
          [-0.2435,  0.5556, -0.2047,  0.4895, -0.2545, -0.3185, -0.2915,
            -0.2842, -0.2099],
          [-0.6780,  2.1815, -0.5836,  1.8659, -0.7740, -0.9894, -0.8962,
            -0.8433, -0.5072],
          [-0.2435,  0.5556, -0.2047,  0.4895, -0.2545, -0.3185, -0.2915,
            -0.2842, -0.2099],
          [-0.5858,  1.2240, -0.4226,  0.8204, -0.7415, -0.8366, -0.7956,
            -0.8764, -0.5525],
          [-0.1256, -0.0217, -0.1516, -0.0996, -0.1516, -0.1776,  0.0303,
            0.0303, -0.0217],
          [-0.0615,  0.4260, -0.1198,  0.4673, -0.1930,  0.4885, -0.0554,
            0.5659, -0.0956]]])
  print(input.unsqueeze(1).shape)
  emb = model(input.unsqueeze(1)).reshape(-1,32)

  #输出形状 (batch_size,32)
  print(emb)
  print(emb.shape)
