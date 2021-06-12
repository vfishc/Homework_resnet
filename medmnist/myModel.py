import torch.nn as nn
import torch

#18层的基础残差结构
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        #downsample下采样函数
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.downsample = downsample

    def forward(self, x):
        origin_x = x
        #如果需要改变x的维度：
        if self.downsample is not None:
            origin_x = self.downsample(x)

        out = nn.ReLU.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += origin_x
        out = nn.ReLU(out)

        return out

#50层的残差结构
class Bottleneck(nn.Module):
    #使用的卷积层的变化为4倍
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        origin_x = x
        if self.downsample is not None:
            origin_x = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += origin_x
        out = self.relu(out)

        return out


class ResNet(nn.Module):
#num_classes训练集的分类个数
    def __init__(self,block,blocks_num,num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channel = 64

        #第一层：卷积层
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)

        self.relu = nn.ReLU(inplace=True)

        #第二层：Maxpooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #各layer，通过make_layer函数实现
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        #layer1中stride默认为1，其余不是
        if stride != 1 or self.in_channel != channel * block.expansion:
            #定义下采样函数，以改变输出维度
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        mylayers = []
        mylayers.append(block(self.in_channel,channel,downsample=downsample,stride=stride,))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            mylayers.append(block(self.in_channel,channel))

        return nn.Sequential(*mylayers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

