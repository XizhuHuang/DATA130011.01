import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, activation='relu', bn=True):
        super().__init__()
        stride = 2 if downsample else 1
        self.activation = self._get_activation(activation)
        self.bn = bn

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False if bn else True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False if bn else True)

        # 根据 bn 参数决定是否使用 BatchNorm
        if self.bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            # 使用 Identity 占位，保持网络结构统一
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            shortcut_layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            ]
            if self.bn:
                shortcut_layers.append(nn.BatchNorm2d(out_channels))
            self.shortcut = nn.Sequential(*shortcut_layers)



    def _get_activation(self, name):
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'swish':
            return nn.SiLU()
        elif name == 'leaky_relu':
            return nn.LeakyReLU(0.1, inplace=True)
        if name == "elu":
            return nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {name}")
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.activation(out)
        return out
    
class ResModel(nn.Module):
    def __init__(self, block_filters=[16,32,64], num_classes=10, activation='relu',dropouts=[0.2,0.3,0.4], gap=True, bn=True):
        super().__init__()
        self.activation = self._get_activation(activation)
        self.gap = gap
        self.bn = bn

        self.conv1 = nn.Conv2d(3, block_filters[0], kernel_size=3, padding=1, bias=not bn)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(block_filters[0])
        else:
            self.bn1 = nn.Identity()  # 占位符
        self.pool = nn.MaxPool2d(2)

        # 动态残差块&dropout
        self.blocks = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        current_channels = block_filters[0]
        feature_size = 16

        for i in range(len(block_filters)):
            out_channels = block_filters[i]
            self.blocks.append(
                ResidualBlock(in_channels=current_channels, out_channels=out_channels, downsample=(current_channels != out_channels), activation=activation, bn=bn)
            )
            self.dropouts.append(nn.Dropout(dropouts[i]))

            if current_channels != out_channels:
                feature_size = feature_size//2
            current_channels = out_channels


        # self.block1 = ResidualBlock(block_filters[0], block_filters[0], activation=activation, bn=bn)
        # self.dropout1 = nn.Dropout(dropouts[0])
        # self.block2 = ResidualBlock(block_filters[0], block_filters[1], downsample=True, activation=activation, bn=bn)
        # self.dropout2 = nn.Dropout(dropouts[1])
        # self.block3 = ResidualBlock(block_filters[1], block_filters[2], downsample=True, activation=activation, bn=bn)
        # self.dropout3 = nn.Dropout(dropouts[2])

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 计算全连接层输入维度
        if gap:
            fc_in = current_channels
        else:
            fc_in = current_channels * feature_size * feature_size  

        self.fc = nn.Linear(fc_in, num_classes)

        # hidden layers
        # self.fc = nn.Sequential(
        #         nn.Linear(fc_in, 128),
        #         self._get_activation(activation), 
        #         nn.Linear(128, num_classes)
        #     )


    def _get_activation(self, name):
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'swish':
            return nn.SiLU()
        elif name == 'leaky_relu':
            return nn.LeakyReLU(0.1, inplace=True)
        elif name == 'elu':
            return nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {name}")
        
    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.pool(x)

        for block, dropout in zip(self.blocks, self.dropouts):
            x = block(x)
            x = dropout(x)

        if self.gap:
            x = self.global_pool(x)
        else:
            # 保持特征图空间维度直接展开
            x = x.view(x.size(0), -1)  # 替代flatten操作
            
        x = torch.flatten(x, 1) if self.gap else x  # 统一处理维度

        x = self.dropouts[-1](x) if len(self.dropouts) > 0 else x
        x = self.fc(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels//ratio),
            nn.SiLU(),
            nn.Linear(channels//ratio, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ =x.size()
        weights = self.squeeze(x).view(b, c)
        weights = self.excitation(weights).view(b,c,1,1)
        return x*weights
    

class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, activation='relu', bn=True, se_ratio=16):
        super().__init__()
        stride = 2 if downsample else 1
        self.activation = self._get_activation(activation)
        self.bn = bn

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=not bn)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not bn)

        if bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            shortcut_layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=not bn)
            ]
            if bn:
                shortcut_layers.append(nn.BatchNorm2d(out_channels))
            self.shortcut = nn.Sequential(*shortcut_layers)
        
        self.se = SEBlock(out_channels, ratio=se_ratio)

    def _get_activation(self, name):
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'swish':
            return nn.SiLU()
        elif name == 'leaky_relu':
            return nn.LeakyReLU(0.1, inplace=True)
        elif name == 'elu':
            return nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {name}")
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += self.shortcut(residual)
        out = self.activation(out)
        return out 


class ResSEModel(nn.Module):
    def __init__(self, block_filters=[16,32,64], num_classes=10, activation='relu', dropouts=[0.2, 0.3, 0.4], gap=True, bn=True, se_ratio=16):
        super().__init__()
        self.activation = self._get_activation(activation)
        self.gap = gap
        self.bn = bn

        self.conv1 = nn.Conv2d(3, block_filters[0], kernel_size=3, padding=1, bias=not bn)
        if bn:
            self.bn1 = nn.BatchNorm2d(block_filters[0])
        else:
            self.bn1 = nn.Identity()
        self.pool = nn.MaxPool2d(2)


        self.block1 = ResidualBlock(block_filters[0], block_filters[0], activation=activation, bn=bn)
        self.dropout1 = nn.Dropout(dropouts[0])

        self.block2 = ResidualBlock(block_filters[0], block_filters[1], downsample=True, activation=activation, bn=bn)
        self.dropout2 = nn.Dropout(dropouts[1])

        self.block3 = ResidualSEBlock(block_filters[1], block_filters[2], downsample=True, activation=activation, bn=bn, se_ratio=se_ratio)
        self.dropout3 = nn.Dropout(dropouts[2])

        # 动态残差块&dropout
        # self.blocks = nn.ModuleList()
        # self.dropouts = nn.ModuleList()

        current_channels = block_filters[0]
        feature_size = 16

        # for i in range(len(block_filters)):
        #     out_channels = block_filters[i]
        #     self.blocks.append(
        #         ResidualSEBlock(in_channels=current_channels, out_channels=out_channels, downsample=(current_channels != out_channels), activation=activation, bn=bn, se_ratio=se_ratio)
        #     )
        #     self.dropouts.append(nn.Dropout(dropouts[i]))

        #     if current_channels != out_channels:
        #         feature_size = feature_size//2
        #     current_channels = out_channels
        

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        fc_in = current_channels if gap else current_channels * feature_size * feature_size

        fc_in = block_filters[2]
        self.fc = nn.Linear(fc_in, num_classes)

        # hidden layers
        # self.fc = nn.Sequential(
        #         nn.Linear(fc_in, 128),
        #         self._get_activation(activation), 
        #         nn.Linear(128, num_classes)
        #     )


    def _get_activation(self, name):
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'swish':
            return nn.SiLU()
        elif name == 'leaky_relu':
            return nn.LeakyReLU(0.1, inplace=True)
        elif name == 'elu':
            return nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # for block, dropout in zip(self.blocks, self.dropouts):
        #     x = block(x)
        #     x = dropout(x)
        x = self.block1(x)
        x = self.dropout1(x)
        x = self.block2(x)
        x = self.dropout2(x)
        x = self.block3(x)

        if self.gap:
            x = self.global_pool(x)
        else:
            x = x.view(x.size(0), -1)

        x = torch.flatten(x, 1) if self.gap else x
        # x = self.dropouts[-1](x) if len(self.dropouts) > 0 else x
        x = self.fc(x)
        return x
    

# ResNet-18 model
def modified_resnet18():
    """加载并修改ResNet18结构适配CIFAR-10"""
    model = models.resnet18(pretrained=False)
    # 修改第一个卷积层（输入通道3，适应32x32尺寸）
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # 移除原maxpool层（避免尺寸过度缩小）
    model.maxpool = nn.Identity()
    # 修改最后的全连接层（输出类别数为10）
    model.fc = nn.Linear(512, 10)
    return model


def classic_resnet18():
    """加载并修改ResNet18结构适配CIFAR-10"""
    model = models.resnet18(pretrained=True)
    # 修改最后的全连接层（输出类别数为10）
    model.fc = nn.Linear(512, 10)
    return model