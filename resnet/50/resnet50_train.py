import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用的设备：", device)
# 定义 ResidualBlock 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.stride != 1 or residual.size(1) != out.size(1):
            residual = self.conv1(residual)
        out += residual
        out = self.relu(out)
        return out

#准备数据集
from torch import nn
from torch.utils.data import DataLoader
train_data=torchvision.datasets.CIFAR10(root="dataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data=torchvision.datasets.CIFAR10(root="dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
#Length长度
train_data_size=len(train_data)
test_data_size=len(test_data)
#如果train_data_size:=10,训练数据集的长度为：10
# print("训练数据集的长度为：{}".format(train_data_size))
# print("测试数据集的长度为：{}".format(test_data_size))
#利用DataLoader来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4)
#创建ResNet50网络模型

class ResNet_50(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet_50, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 256, 3, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 512, 4, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 1024, 6, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 2048, 3, stride=2)
        self.fc = nn.Linear(512 * 4, num_classes)
        # **************************

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):  # 3*32*32
        out = self.conv1(x)  # 64*32*32
        out = self.layer1(out)  # 64*32*32
        out = self.layer2(out)  # 128*16*16
        out = self.layer3(out)  # 256*8*8
        out = self.layer4(out)  # 512*4*4
        out = F.avg_pool2d(out, 4)  # 512*1*1
        # print(out.size())
        out = out.view(out.size(0), -1)  # 512
        out = self.fc(out)
        return out

res50 = ResNet_50(ResidualBlock)
res50 = res50.to(device)

#损失函数
Loss_fn=nn.CrossEntropyLoss()
#优化器
Learning_rate=0.01
optimizer=torch.optim.SGD(res50.parameters(),lr=Learning_rate)

#设置训练网络的一些参数
#记录训练的次数
total_train_step =0
#记录测试的次数
total_test_step =0
# 每20轮储存一次模型参数
save_interval = 20
#训练的轮数
epoch=100
#添加tensorboard
writer=SummaryWriter('res50/logs_resnet50')
if __name__ == '__main__':
    for i in range(epoch):
        print("----第{}轮训练开始-----".format(i + 1))

        # 训练步骤开始
        res50.train()
        for data in train_dataloader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = res50(imgs)
            loss = Loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_step += 1
            if total_train_step % 1000 == 0:
                print("训练次数：{}，Loss:{}".format(total_train_step, loss.item()))
                writer.add_scalar('train_loss', loss.item(), total_train_step)

        # 测试步骤开始
        res50.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in test_dataloader:
                imgs, targets = data
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = res50(imgs)
                loss = Loss_fn(outputs, targets)
                total_test_loss += loss.item()
                accuracy = (outputs.argmax(1) == targets).sum().item()
                total_accuracy += accuracy

        # 计算平均测试损失和准确率
        avg_test_loss = total_test_loss / len(test_dataloader)
        avg_accuracy = total_accuracy / test_data_size
        print("整体测试集上的Loss: {}".format(avg_test_loss))
        print("整体测试集上的正确率: {}".format(avg_accuracy))

        writer.add_scalar("test_loss", avg_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", avg_accuracy, total_test_step)
        total_test_step += 1
        # 储存模型参数
        if (i + 1) % save_interval == 0:
            torch.save(res50, "res50_{}.pth".format(i))

writer.close()
torch.cuda.empty_cache()  # 清理GPU内存