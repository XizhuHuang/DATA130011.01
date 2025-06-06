import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

def test_model(model_type, conv_configs=None, fc_configs=None, use_global_avg_pool=None, model_path=None):
    if model_type == 'cnn':
        model = nn.models.Model_CNN(
        conv_configs=conv_configs,
        fc_configs=fc_configs,
        act_func='ReLU',
        use_global_avg_pool=use_global_avg_pool 
    )         
    elif model_type == "mlp":
        model = nn.models.Model_MLP()
        
    model.load_model(model_path)
    
    test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
    test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

    with gzip.open(test_images_path, 'rb') as f:
            magic, num, rows, cols = unpack('>4I', f.read(16))
            test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
        
    with gzip.open(test_labels_path, 'rb') as f:
            magic, num = unpack('>2I', f.read(8))
            test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    test_imgs = test_imgs / test_imgs.max()

    if model_type == 'cnn':
        test_imgs = test_imgs.reshape(-1, 1, 28, 28)  # 转换为[N, C, H, W]格式
        # 在test_model函数中添加以下检查：
        print("Test images shape:", test_imgs.shape)  # 应输出 (10000, 1, 28, 28)
        print("Test images max/min:", test_imgs.max(), test_imgs.min())  # 应为 1.0 和 0.0

    
    # 选取前5个样本手动预测
    sample = test_imgs[:10]
    logits = model(sample)
    preds = np.argmax(logits, axis=1)
    print("Sample predictions:", preds)
    print("True labels:", test_labs[:10])

    logits = model(test_imgs)
    return nn.metric.accuracy(logits, test_labs)


conv_configs = [
    {
        'type': 'conv',
        'in_channels': 1,          # MNIST灰度图通道数
        'out_channels': 16,       
        'kernel_size': 5,
        'stride': 1,
        'padding': 0,
        'weight_decay': True,      # 可选：添加正则化
        'weight_decay_lambda': 1e-4
    },
    {
        'type': 'pool',
        'pool_type': 'avg',
        'kernel_size': 2,
        'stride': 2
    },
    {
        'type': 'conv',
        'in_channels': 16,          # MNIST灰度图通道数
        'out_channels': 32,       
        'kernel_size': 5,
        'stride': 1,
        'padding': 0,
        'weight_decay': True,      # 可选：添加正则化
        'weight_decay_lambda': 1e-4
    },
    {
        'type': 'pool',
        'pool_type': 'avg',
        'kernel_size': 2,
        'stride': 2
    }
]

fc_configs = [
    (512, 10)  
]


print('CNN model')
print(test_model(model_type='cnn', conv_configs=conv_configs, fc_configs=fc_configs, use_global_avg_pool=False, model_path=r'.\saved_models\cnn_model\best_model.pickle'))

print('Multi-Layer model')
print(test_model(model_type="mlp", model_path=r'.\saved_models\mlp_model\best_model.pickle'))