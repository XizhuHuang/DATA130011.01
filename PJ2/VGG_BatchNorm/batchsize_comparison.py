import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from tqdm import tqdm
from VGG_Loss_Landscape import set_random_seeds, train, get_dataloader
from models.vgg import VGG_A, VGG_A_BatchNorm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 不同批量大小
batch_sizes = [16, 32, 64, 128]
epochs_n = 20  # 训练轮数
learning_rate = 0.001  # 固定学习率

# 存储结果的数据结构
results = {
    'vgg_a': {
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': []
    },
    'vgg_a_bn': {
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': []
    }
}

# 训练不同批量大小的模型
for batch_size in batch_sizes:
    print(f"\n{'='*50}")
    print(f"开始训练批量大小: {batch_size}")
    print(f"{'='*50}")
    
    # 获取数据加载器
    train_loader, val_loader = get_dataloader(batch_size)
    
    # 设置随机种子保证可复现性
    set_random_seeds(2020, device)
    
    # 训练VGG-A模型
    model_vgg_a = VGG_A().to(device)
    optimizer_vgg_a = torch.optim.Adam(model_vgg_a.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"\n训练VGG-A (批量大小: {batch_size})")
    train_losses, val_losses, train_accuracies, val_accuracies = train(
        model_vgg_a, optimizer_vgg_a, criterion, train_loader, val_loader, epochs_n=epochs_n
    )
    
    # 存储结果
    results['vgg_a']['train_losses'].append(train_losses)
    results['vgg_a']['val_losses'].append(val_losses)
    results['vgg_a']['train_accuracies'].append(train_accuracies)
    results['vgg_a']['val_accuracies'].append(val_accuracies)
    
    # 训练VGG-A-BatchNorm模型
    model_vgg_bn = VGG_A_BatchNorm().to(device)
    optimizer_vgg_bn = torch.optim.Adam(model_vgg_bn.parameters(), lr=learning_rate)
    
    print(f"\n训练VGG-A-BN (批量大小: {batch_size})")
    train_losses_bn, val_losses_bn, train_accuracies_bn, val_accuracies_bn = train(
        model_vgg_bn, optimizer_vgg_bn, criterion, train_loader, val_loader, epochs_n=epochs_n
    )
    
    # 存储结果
    results['vgg_a_bn']['train_losses'].append(train_losses_bn)
    results['vgg_a_bn']['val_losses'].append(val_losses_bn)
    results['vgg_a_bn']['train_accuracies'].append(train_accuracies_bn)
    results['vgg_a_bn']['val_accuracies'].append(val_accuracies_bn)


# 创建图形
plt.figure(figsize=(20, 10))

for i, batch_size in enumerate(batch_sizes):
    # 绘制最大最小值曲线
    plt.subplot(2, 4, i+1)  
    plt.plot(results['vgg_a_bn']['train_losses'][i], 'b', label='VGG-A-BN') 
    plt.plot(results['vgg_a']['train_losses'][i], 'r', label='VGG-A') 
    plt.title(f'Train Loss (batch_size={batch_size})')
    plt.xlabel('Steps') 
    plt.ylabel('Loss') 
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12) 

    plt.subplot(2, 4, i+5)  
    plt.plot(results['vgg_a']['val_accuracies'][i], 'r', label='VGG-A') 
    plt.plot(results['vgg_a_bn']['val_accuracies'][i], 'b', label='VGG-A-BN') 
    plt.title(f'Test Acc (batch_size={batch_size})') 
    plt.xlabel('Epochs')  
    plt.ylabel('Accuracy(%)') 
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12) 

plt.tight_layout()
# 保存图像
fig_dir = 'figures'
os.makedirs(fig_dir, exist_ok=True)
plt.savefig(os.path.join(fig_dir, 'batchsize_comparison.png'), dpi=300)

print("训练完成！图像已保存至figures目录")
# 显示图表
plt.show()