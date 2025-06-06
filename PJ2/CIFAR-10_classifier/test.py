import argparse
import torch
from torch import nn
from data_loader import get_data_loaders
from models import ResModel

def test(model, device, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)
    return avg_loss, acc

def main():
    parser = argparse.ArgumentParser(description='ResModel Testing')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--filters', type=int, nargs=3, default=[16, 32, 64])
    parser.add_argument('--activation', choices=['swish', 'relu', 'leaky_relu', 'elu'], default='swish')
    parser.add_argument('--gap', action='store_true', help='Use GlobalAvgPool')
    parser.add_argument('--no-gap', dest='gap', action='store_false', help='Use flatten')
    parser.set_defaults(gap=True)
    parser.add_argument('--bn', action='store_true', help='Use BatchNorm')
    parser.add_argument('--no-bn', dest='bn', action='store_false', help='No BatchNorm')
    parser.set_defaults(bn=True)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = ResModel(
        block_filters=args.filters,
        activation=args.activation,
        gap=args.gap
    ).to(device)
    
    # 加载训练好的权重
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # 获取测试集
    _, test_loader = get_data_loaders(args.batch_size)

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc = test(model, device, test_loader, criterion)
    
    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}%")

if __name__ == '__main__':
    main()