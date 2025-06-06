import torch
from torchvision import datasets, transforms

def get_data_loaders(batch_size=128, data_dir='./data'):
    # data augmentation and nomalization
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    # ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.05),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # 平移增强
        # transforms.RandomRotation(15),  # 旋转增强
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(p=0.5)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root=data_dir, train=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader

