import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import time
from collections import defaultdict
from data_loader import get_data_loaders
from models import ResModel, ResSEModel, classic_resnet18, modified_resnet18
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, LambdaLR


def train(model, device, train_loader, criterion, optimizer, epoch, reg_type, lambda_):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        original_target = target.clone()

        # 根据损失函数类型转换target
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            # 转换为one-hot编码
            target = nn.functional.one_hot(target, num_classes=10).float()
        elif isinstance(criterion, nn.MSELoss):
            # 转换为one-hot并归一化
            target = nn.functional.one_hot(target, num_classes=10).float() / 10.0

        optimizer.zero_grad()
        output = model(data)

        # MSE需要特殊处理输出
        if isinstance(criterion, nn.MSELoss):
            output = torch.softmax(output, dim=1)
            # output = output / output.sum(dim=1, keepdim=True)  # 确保概率归一化

        loss = criterion(output, target)

        if reg_type == 'l2':
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += lambda_ * l2_reg
        elif reg_type == 'l1':
            l1_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l1_reg += torch.norm(param, p=1)
            loss += lambda_ * l1_reg

        loss.backward()
        optimizer.step()

        # 准确率计算适配不同损失函数
        if isinstance(criterion, (nn.BCEWithLogitsLoss, nn.MSELoss)):
            pred = output.argmax(dim=1) 
        else:
            pred = output.argmax(dim=1, keepdim=True)
            
        correct += pred.eq(original_target.view_as(pred)).sum().item()
        total_loss += loss.item()


    avg_loss = total_loss / len(train_loader)
    acc = 100.*correct / len(train_loader.dataset)
    return avg_loss, acc


def test(model, device, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            original_target = target.clone()

            # 转换目标格式
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                target = nn.functional.one_hot(target, 10).float()
            elif isinstance(criterion, nn.MSELoss):
                target = nn.functional.one_hot(target, 10).float() / 10.0

            output = model(data)
            
            # 处理MSE输出
            if isinstance(criterion, nn.MSELoss):
                output = torch.softmax(output, dim=1)

            loss = criterion(output, target)
            total_loss += loss.item()

            # 计算预测结果
            if isinstance(criterion, (nn.BCEWithLogitsLoss, nn.MSELoss)):
                pred = output.argmax(dim=1)
            else:
                pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(original_target.view_as(pred)).sum().item()

    avg_loss = total_loss / len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)
    return avg_loss, acc



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save best model')
    parser.add_argument('--log_dir', type=str, default=None, help='Directory to save training log')
    parser.add_argument('--model', choices=['resmodel', 'classic_resnet18', 'modified_resnet18', 'semodel'], default='resmodel')
    parser.add_argument('--ratio', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--optimizer', choices=['sgd', 'adam', 'adagrad', 'rmsprop'], default='sgd')
    parser.add_argument('--scheduler', choices=['multistep', 'cosine', 'linear'], default='cosine')
    parser.add_argument('--loss_type', choices=['ce', 'multimargin', 'bce', 'mse'], default='ce', help='Loss function type')
    parser.add_argument('--activation', choices=['swish', 'relu', 'leaky_relu', 'elu'], default='swish')
    parser.add_argument('--gap', action='store_true', help='Use GlobalAvgPool')
    parser.add_argument('--no-gap', dest='gap', action='store_false', help='Use flatten')
    parser.set_defaults(gap=True)
    parser.add_argument('--bn', action='store_true', help='Use BatchNorm')
    parser.add_argument('--no-bn', dest='bn', action='store_false', help='No BatchNorm')
    parser.set_defaults(bn=True)
    parser.add_argument('--reg_type', choices=['none', 'l1', 'l2'], default='l2',  help='Regularization type')
    parser.add_argument('--reg_lambda', type=float, default=0.0001, help='Regularization coefficient for L1/L2')
    # filters参数为动态长度
    parser.add_argument('--filters', type=int, nargs='+', default=[128, 256, 512], help='List of filter numbers for each residual block (length must match num_blocks)')
    parser.add_argument('--dropouts', type=float, nargs='+', default=[0.2, 0.3, 0.4])
    parser.add_argument('--patience', type=int, default=12, help='Number of epochs to wait before early stopping if no improvement')
    args = parser.parse_args()
        
    # 参数校验
    if len(args.dropouts) != len(args.filters):
        raise ValueError("Numder of filters must match number of dropouts.")

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_data_loaders(args.batch_size)

    # model
    if args.model == 'classic_resnet18':
        model = classic_resnet18().to(device)
    elif args.model == 'modified_resnet18': 
        model = modified_resnet18().to(device)
    elif args.model == 'resmodel': 
        model = ResModel(block_filters=args.filters, activation=args.activation, dropouts=args.dropouts, gap=args.gap, bn=args.bn).to(device)
    elif args.model == 'semodel': 
        model = ResSEModel(block_filters=args.filters, activation=args.activation, dropouts=args.dropouts, gap=args.gap, bn=args.bn, se_ratio=args.ratio).to(device)

    criterion = nn.CrossEntropyLoss()

    # loss function
    if args.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_type == 'multimargin':
        criterion = nn.MultiMarginLoss()
    elif args.loss_type == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss_type == 'mse':
        criterion = nn.MSELoss()

    # optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)


    # scheduler
    if args.scheduler == 'multistep':
        scheduler = MultiStepLR(optimizer, milestones=[10,20,30,40,50,60], gamma=0.5)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.001)
    elif args.scheduler == 'linear':
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch/args.epochs)

    # 时间记录容器
    time_records = defaultdict(list)
    total_start_time = time.time()

    # 训练数据记录容器
    train_loss_records = defaultdict(list)
    train_acc_records = defaultdict(list)
    val_loss_records = defaultdict(list)
    val_acc_records = defaultdict(list)

    best_acc = 0.0
    # early stopping
    best_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(1, args.epochs+1):
        epoch_start = time.time()
        train_loss, train_acc = train(model, device, train_loader, criterion, optimizer, epoch, args.reg_type, args.reg_lambda)
        val_loss, val_acc = test(model, device, test_loader, criterion)

        train_loss_records['epoch_tloss'].append(train_loss)
        train_acc_records['epoch_tacc'].append(train_acc)
        val_loss_records['epoch_vloss'].append(val_loss)
        val_acc_records['epoch_vacc'].append(val_acc)

        # 记录epoch耗时
        epoch_time = time.time() - epoch_start
        time_records['epoch_times'].append(epoch_time)

        scheduler.step()

        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% |'
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% |'
              f'Time: {epoch_time:.2f}s')
              
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            if args.save_dir is not None:
                torch.save(model.state_dict(), 
                      os.path.join(args.save_dir, 'best_model.pth'))
        else:
            epochs_no_improve += 1


        if epochs_no_improve >= args.patience:
            print(f'\nEarly stopping triggered at epoch {epoch}! No improvement for {args.patience} epochs.')
            early_stop = True
            break
            
    # 计算总耗时
    total_time = time.time() - total_start_time
    avg_epoch_time = sum(time_records['epoch_times'])/len(time_records['epoch_times'])

    # 生成时间报告
    time_report = f"""
    ======== Training Time Summary ========
    Total epochs:          {args.epochs}
    Total training time:   {total_time:.2f} seconds
    Average epoch time:    {avg_epoch_time:.2f} seconds
    Fastest epoch:         {min(time_records['epoch_times']):.2f}s @ epoch {
        time_records['epoch_times'].index(min(time_records['epoch_times']))+1}
    Slowest epoch:         {max(time_records['epoch_times']):.2f}s @ epoch {
        time_records['epoch_times'].index(max(time_records['epoch_times']))+1}
    """
    print(time_report)


    # 日志保存
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_path = os.path.join(args.log_dir, 'train_log.txt')
        
        with open(log_path, 'w') as f:
            # 超参数
            f.write("="*40 + " Hyperparameters " + "="*40 + "\n")
            for key, value in vars(args).items():
                f.write(f"{key:20}: {value}\n")
            
            # 时间总结
            f.write("\n" + "="*40 + " Time Summary " + "="*40 + "\n")
            f.write(time_report)
            
            f.write("\n\n" + "="*40 + " Detailed Metrics " + "="*40 + "\n")
            f.write(f"{'Epoch':<6} {'Train Loss':<10} {'Train Acc':<10} {'Val Loss':<10} {'Val Acc':<10} {'Time(s)':<10}\n")
            f.write("-"*65 + "\n")
            
            # 逐epoch写入数据
            for epoch_idx in range(len(train_loss_records['epoch_tloss'])):
                line = (
                    f"{epoch_idx+1:<6} "
                    f"{train_loss_records['epoch_tloss'][epoch_idx]:<10.4f} "
                    f"{train_acc_records['epoch_tacc'][epoch_idx]:<10.2f} "
                    f"{val_loss_records['epoch_vloss'][epoch_idx]:<10.4f} "
                    f"{val_acc_records['epoch_vacc'][epoch_idx]:<10.2f} " 
                    f"{time_records['epoch_times'][epoch_idx]:<10.2f}\n"
                )
                f.write(line)
            
            # 最佳结果
            f.write("\n" + "="*40 + " Best Result " + "="*40 + "\n")
            f.write(f"Best Validation Accuracy: {best_acc:.2f}%\n")
                
if __name__ == '__main__':
    main()
