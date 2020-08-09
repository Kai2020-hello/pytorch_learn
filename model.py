from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim


# 定义网络

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 两个卷积
        self.conv1 = nn.Conv2d(1,6,5) # 输入通道是1 输出通道是16 kernal大小是5*5
        self.conv2 = nn.Conv2d(6,16,5)

        self.fc1 = nn.Linear(16*5*5,128)
        self.fc2 = nn.Linear(128,54)
        self.fc3 = nn.Linear(54,10)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.reshape(-1,self.num_flat_feature(x))
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_feature(self,x):
        sizes = x.size()[1:]
        feature_num = 1
        for size in sizes:
            feature_num *= size
        return feature_num

net = Net()

print(net)

param = list(net.parameters())
print(len(param))

print(param[4].size())

# 定义输入 前向传播
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# 定义真值
target = torch.rand(10)
target = target.reshape(1,-1)

#定义loss函数
cer = nn.MSELoss()

#计算loss
loss = cer(out,target)

print(loss)
print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


# 反向传播
#net.zero_grad()

#print(net.conv1.bias.grad)

#loss.backward()

#print(net.conv1.bias.grad)


# 更新权重
lr = 0.001
#for f in net.parameters():
#    f.data.sub_(lr * f.grad.data)



# 更新权重过程使用优化器搞定

opt = optim.SGD(net.parameters(),lr)

opt.zero_grad()
out = net(input)
loss = cer(out,target)
loss.backward()
opt.step()
