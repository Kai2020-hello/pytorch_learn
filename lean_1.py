from __future__ import print_function
import torch

## 张量生成

x = torch.empty(3,5)
print(x)

## 随机生成一个张量

x = torch.rand(3,5)
print(x)


#生成一个全为0 类型为long的张量

print(x)

# 直接从数据中生成

x = torch.tensor([2,4,5])
print(x)

# 加法 ：对应元素相加
x = torch.zeros(3,5,dtype=torch.long)
y = x.new_ones(3,5)

z = x + y

print(z)
# 原位操作 后面是_的操作   类似+= 
print(z.t_().size())

#改变形状
z = z.reshape(15)
print(z)

# 切片
print(x[2:,3:])

# 转成numpy
print(x.numpy())


## 设备转移

device = torch.device("cpu")
print(device)

x.to(device)
