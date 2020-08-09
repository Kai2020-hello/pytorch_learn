# pytorch 默认是单Gpu,使用to方法就可以把模型，数据搬迁到gpu但是，只是把这个tensor的副本

import torch 
import torchvision
from torch.utils.data import Dataset,DataLoader

