import torch 
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # transforms加载的数据是0，1之间的pilimage ，需要把数据转化成 -1，1之间的数据
    ]
)

#加载数据

trainset = torchvision.datasets.CIFAR10(root="./data",train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)


testset = torchvision.datasets.CIFAR10(root="./data",train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=True,num_workers=2)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



import matplotlib.pyplot as plt
import numpy as np

# 输出图像的函数
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 随机获取训练图片
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 显示图片
imshow(torchvision.utils.make_grid(images))
# 打印图片标签
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))




#定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 两个卷积
        self.conv1 = nn.Conv2d(3,6,5) # 输入通道是1 输出通道是16 kernal大小是5*5
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

#定义损失函数和优化器
lr = 0.0001
cer = torch.nn.MSELoss()
opt = torch.optim.SGD(net.params(),lr=lr,momentum=0.9)


## 训练网络
for epoch in range(2):
    runing_loss = 0.0
    for i,data in enumerate(trainloader,0):
        #数据
        inputes ,label = data

        opt.zero_grad()

        #forwrd 
        out = net(input)
        loss = cer(out,label)

        #backward
        loss.backward()
        opt.step()

        #打印信息
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


#模型保存
path = './cifar_net.pth'
torch.save(net.state_dict(),path)


#加载模型
net = Net()
net.load_state_dict(torch.load(path))


#加载测试训练集
dataiter = iter(testloader)
images, labels = dataiter.next()

#预估数据
outputs = net(images)
_,predicts = torch.max(outputs,1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))