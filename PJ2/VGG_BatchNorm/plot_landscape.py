import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.vgg import VGG_A, VGG_A_BatchNorm
from VGG_Loss_Landscape import set_random_seeds, train
from data.loaders import get_cifar_loader

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机种子
set_random_seeds(seed_value=2020, device=device)

# 初始化数据加载器
train_loader = get_cifar_loader(train=True, batch_size=128)
val_loader = get_cifar_loader(train=False, batch_size=128)

# 学习率列表
learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]
epochs_n = 20  # 训练轮数

# 存储所有训练损失
VGG_A_all_losses = []  # 存储每个学习率下VGG_A的训练损失
VGG_A_BN_all_losses = []  # 存储每个学习率下VGG_A_BN的训练损失

# 函数：保存日志到文件
def save_logs(model_name, lr, train_losses, val_losses, train_accuracies, val_accuracies):
    """保存训练日志到文件"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'{model_name}_lr{lr}.txt')
    
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Training Logs for {model_name} with lr={lr}\n")
        log_file.write("=" * 60 + "\n")
        log_file.write("Epoch\tTrain Loss\tVal Loss\tTrain Acc\tVal Acc\n")
        
        # 写入每个epoch的统计数据
        for epoch in range(epochs_n):
            log_file.write(f"{epoch+1}\t{train_losses[epoch]:.6f}\t{val_losses[epoch]:.6f}\t{train_accuracies[epoch]:.2f}%\t{val_accuracies[epoch]:.2f}%\n")

# 训练模型并收集损失数据
print("开始训练模型...")
for lr in tqdm(learning_rates, desc="不同学习率训练进度", leave=True):
    # 训练VGG_A模型
    model_vgg_a = VGG_A().to(device)
    optimizer_vgg_a = torch.optim.Adam(model_vgg_a.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 训练并获取每个batch的损失
    train_losses, val_losses, train_accuracies, val_accuracies = train(
        model_vgg_a, optimizer_vgg_a, criterion, train_loader, val_loader, epochs_n=epochs_n
    )
    
    # 保存日志
    save_logs('vgg_a', lr, train_losses, val_losses, train_accuracies, val_accuracies)
    
    # 存储训练损失（每个batch的损失）
    VGG_A_all_losses.append(train_losses)
    
    # 训练VGG_A_BatchNorm模型
    model_vgg_bn = VGG_A_BatchNorm().to(device)
    optimizer_vgg_bn = torch.optim.Adam(model_vgg_bn.parameters(), lr=lr)
    
    # 训练并获取每个batch的损失
    train_losses_bn, val_losses_bn, train_accuracies_bn, val_accuracies_bn = train(
        model_vgg_bn, optimizer_vgg_bn, criterion, train_loader, val_loader, epochs_n=epochs_n
    )
    
    # 保存日志
    save_logs('vgg_a_bn', lr, train_losses_bn, val_losses_bn, train_accuracies_bn, val_accuracies_bn)
    
    # 存储训练损失（每个batch的损失）
    VGG_A_BN_all_losses.append(train_losses_bn)

# 转换为NumPy数组以便计算
VGG_A_all_losses = np.array(VGG_A_all_losses)
VGG_A_BN_all_losses = np.array(VGG_A_BN_all_losses)

# 计算每个迭代步骤的最大值和最小值
VGG_A_min = np.min(VGG_A_all_losses, axis=0)
VGG_A_max = np.max(VGG_A_all_losses, axis=0)
VGG_A_BN_min = np.min(VGG_A_BN_all_losses, axis=0)
VGG_A_BN_max = np.max(VGG_A_BN_all_losses, axis=0)

# 下采样以减少绘图点数量（每30个点取一个）
sampling_step = 30
sampled_indices = np.arange(0, len(VGG_A_min), sampling_step)

VGG_A_min_sampled = VGG_A_min[sampled_indices]
VGG_A_max_sampled = VGG_A_max[sampled_indices]
VGG_A_BN_min_sampled = VGG_A_BN_min[sampled_indices]
VGG_A_BN_max_sampled = VGG_A_BN_max[sampled_indices]

# 创建图形
plt.figure(figsize=(12, 8))

# 填充VGG-A的损失范围
plt.fill_between(sampled_indices, VGG_A_min_sampled, VGG_A_max_sampled, 
                 color='red', alpha=0.2, label='VGG-A Loss Range')

# 填充VGG-A-BN的损失范围
plt.fill_between(sampled_indices, VGG_A_BN_min_sampled, VGG_A_BN_max_sampled, 
                 color='blue', alpha=0.2, label='VGG-A-BN Loss Range')

# 绘制最大最小值曲线
plt.plot(sampled_indices, VGG_A_min_sampled, 'r', linewidth=1, alpha=0.7)
plt.plot(sampled_indices, VGG_A_max_sampled, 'r', linewidth=1, alpha=0.7)
plt.plot(sampled_indices, VGG_A_BN_min_sampled, 'b', linewidth=1, alpha=0.7)
plt.plot(sampled_indices, VGG_A_BN_max_sampled, 'b', linewidth=1, alpha=0.7)

# 添加标题和标签
plt.title("Loss Landscape Comparison: VGG-A vs VGG-A-BN", fontsize=16)
plt.xlabel("Training Iterations (Steps)", fontsize=12)
plt.ylabel("Training Loss", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# 添加图例
plt.legend(fontsize=12)

# 调整布局
plt.tight_layout()

# 保存图像
fig_dir = 'figures'
os.makedirs(fig_dir, exist_ok=True)
plt.savefig(os.path.join(fig_dir, 'loss_landscape_comparison.png'), dpi=300)

print("训练完成！损失景观图已保存至figures目录")
plt.show()

