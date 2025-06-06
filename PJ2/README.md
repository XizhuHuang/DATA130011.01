# 📘 PJ2 

## 📑 目录（Table of Contents）

* [1. Task1：CIFAR-10 图像分类项目](#1-task1cifar-10-图像分类项目)

  * [1.1 项目简介](#11-项目简介)
  * [1.2 代码结构](#12-代码结构)
  * [1.3 使用说明](#13-使用说明)
  * [1.4 最佳模型权重下载](#14-最佳模型权重下载)

* [2. Task2：VGG-A 与 Batch Normalization 实验项目](#2-task2vgg-a-与-batch-normalization-实验项目)

  * [2.1 项目结构](#21-项目结构)
  * [2.2 主要实验内容与结论](#22-主要实验内容与结论)
  * [2.3 模型权重下载](#23-模型权重下载)
  * [2.4 快速开始](#24-快速开始)
  * [2.5 依赖环境](#25-依赖环境)


## 1. Task1：CIFAR-10 图像分类项目

### 1.1 项目简介

本项目聚焦于 CIFAR-10 数据集的图像分类任务，通过设计和评估一系列模型架构与训练策略，不断提升模型性能。我们逐步引入更大容量的网络结构、强化数据增强技术、先进的特征聚合方法、通道注意力机制（SE模块）以及精心调优的超参数，最终成功构建出性能卓越的 SE-ResNet 模型。

该模型在最佳配置下于测试集上取得了 **93.04%** 的准确率，展现出良好的泛化能力。


### 1.2 代码结构

```
CIFAR-10_classifier/
├── __pycache__/
├── figures/
├── log/            # 包含所有实验的训练日志
├── data_loader.py  # 数据加载模块
├── models.py       # 模型定义
├── test.py         # 测试脚本
├── train.py        # 训练脚本
└── vis.ipynb       # 可视化脚本
```


### 1.3 使用说明

#### 环境依赖

* Python 3.7+
* PyTorch 1.8+
* torchvision
* 其他标准 Python 库（argparse, time, collections）

#### 数据准备

运行以下脚本自动下载 CIFAR-10 数据集：

```bash
python data_loader.py
```

#### 训练模型

```bash
python train.py \
  --model semodel \
  --epochs 100 \
  --batch_size 128 \
  --lr 0.1 \
  --optimizer sgd \
  --scheduler cosine \
  --loss_type ce \
  --activation swish \
  --gap \
  --bn \
  --reg_type l2 \
  --reg_lambda 0.0001 \
  --filters 128 256 512 \
  --dropouts 0 0 0 \
  --patience 12 \
  --save_dir ./checkpoints \
  --log_dir ./log
```

参数说明：

* `--model semodel`：选择带SE模块的ResNet模型。
* `--epochs 100`：训练100个epoch。
* `--batch_size 128`：每批次128张图片。
* `--lr 0.1`：初始学习率为0.1。
* `--optimizer sgd`：使用动量SGD优化器。
* `--scheduler cosine`：采用余弦退火学习率调度。
* `--loss_type ce`：交叉熵损失。
* `--activation swish`：Swish激活函数。
* `--gap`：使用全局平均池化层。
* `--bn`：使用批归一化。
* `--reg_type l2` 和 `--reg_lambda 0.0001`：使用L2正则化，系数0.0001。
* `--filters 128 256 512`：每个残差块的通道数。
* `--dropouts 0 0 0`：对应各残差块的Dropout概率。
* `--patience 12`：早停等待12个epoch无提升。
* `--save_dir` 和 `--log_dir`：分别指定模型保存和日志输出路径。


#### 测试模型

```bash
python test.py \
  --model_path ./checkpoints/best_model.pth \
  --batch_size 256 \
  --filters 128 256 512 \
  --activation swish \
  --gap \
  --bn
```


### 1.4 最佳模型权重下载

百度网盘链接：
🔗 [https://pan.baidu.com/s/1Uw6xB8R3UUhc0v8qa8UTYg](https://pan.baidu.com/s/1Uw6xB8R3UUhc0v8qa8UTYg)
🔐 提取码：`fw7d`

下载后请放入 `checkpoints/` 或指定路径下。


## 2. Task2：VGG-A 与 Batch Normalization 实验项目

### 2.1 项目结构

```
VGG_BatchNorm/
├── __pycache__/             
├── data/                    # 数据集
├── figures/                 # 图表输出
├── logs/                    # 日志文件
├── models/                  # 模型定义（VGG-A, VGG-A-BN）
├── utils/                   # 工具函数
├── batchsize_comparison.py  # 不同批量训练实验
├── plot_landscape.py        # Loss Landscape 可视化
└── VGG_Loss_Landscape.py    # Landscape 核心代码
```


### 2.2 主要实验内容与结论

* **基础性能对比**
  VGG-A-BN 显著提升了训练稳定性、收敛速度和测试准确率。

* **Loss Landscape 可视化**
  不同学习率下，VGG-A-BN 展现更小的 loss 波动区间，训练过程更加平稳。

* **不同批量大小实验**
  在 batch size = 16\~128 的设置下，BN 有效缓解了梯度不稳定，提升了小批训练的可行性和模型泛化能力。


### 2.3 模型权重下载

百度网盘链接：
🔗 [https://pan.baidu.com/s/1Uw6xB8R3UUhc0v8qa8UTYg](https://pan.baidu.com/s/1Uw6xB8R3UUhc0v8qa8UTYg)
🔐 提取码：`fw7d`

请放置于 `saved_models/` 目录下。


### 2.4 快速开始

```bash
# 克隆仓库
git clone https://github.com/XizhuHuang/DATA130011.01/tree/main/PJ2/VGG_BatchNorm
cd VGG_BatchNorm

# 运行不同 batch size 的训练实验
python batchsize_comparison.py

# 可视化 Loss Landscape
python plot_landscape.py
```



### 2.5 依赖环境

* Python 3.8+
* PyTorch 1.10+
* torchvision
* numpy
* matplotlib
* tqdm




