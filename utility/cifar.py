import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from utility.cutout import Cutout


class Cifar10:
    def __init__(self, batch_size, threads):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)
        
        # Valid dataset for finetuning
        valid_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=test_transform)
        num_train = len(train_set)
        indices = list(range(num_train))
        split = int(np.floor(0.1 * num_train))

        np.random.seed(42)
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=threads, sampler=train_sampler)
        self.valid = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, num_workers=threads, sampler=valid_sampler)
        
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])


class Cifar100:
    def __init__(self, batch_size, threads):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=train_transform)
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=test_transform)
               
        # Valid dataset for finetuning
        valid_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=test_transform)
        num_train = len(train_set)
        indices = list(range(num_train))
        split = int(np.floor(0.1 * num_train))

        np.random.seed(42)
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=threads, sampler=train_sampler)
        self.valid = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, num_workers=threads, sampler=valid_sampler)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

        # self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR10(root='./dataa', train=True, download=False, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])