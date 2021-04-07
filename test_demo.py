from main import *

# _, _ = main(dataname = 'FashionMNIST', ood_dataname=['notMNIST', 'MNIST'], gamma=0.1, network='resnet20', grad_clip= 1, n_epochs=50)

_, _ = main(dataname = 'CIFAR10', ood_dataname=['SVHN', 'CIFAR10', 'LSUN'], gamma=0.1, network='resnet20', grad_clip= 1, n_epochs=50)

# _, _ = main(dataname = 'SVHN', ood_dataname=['CIFAR10', 'CIFAR10', 'LSUN'], gamma=0.1, network='resnet20', grad_clip= 1, n_epochs=50)
