from model.resnet import resnet56
from model.wide_res_net import WideResNet
from model.densenet import DenseNet121
from model.vgg import vgg19_bn
from utility.sam import SAM
from utility.cifar import Cifar10, Cifar100


def get_dataset(args):
    if args.dataset == 'cifar10':
        dataset = Cifar10(args.batch_size, args.threads)
    elif args.dataset == 'cifar100':
        dataset = Cifar100(args.batch_size, args.threads)
    return dataset


def get_optim(model, base_optimizer, args):
    if args.adam:
        kwargs= {'lr':args.learning_rate, "weight_decay": args.weight_decay}
    else:
        kwargs= {'lr':args.learning_rate, "momentum":args.momentum, "weight_decay": args.weight_decay}
    
    if args.optim == 'base':
        optimizer = base_optimizer(model.parameters(), **kwargs)
    elif args.optim in ['sam', 'bisam_log', 'bisam_tanh']:
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, **kwargs)
    else:
        raise TypeError('Optimizer is wrong.')
    return optimizer


def get_model(args, device):
    if args.dataset == 'cifar10':
        num_classes = 10  
    elif args.dataset == 'cifar100':
        num_classes = 100
        
    if args.model == 'wrn2810':
        model = WideResNet(28, 10, args.dropout, in_channels=3, labels=num_classes).to(device)
    elif args.model == 'wrn282':
        model = WideResNet(28, 2, args.dropout, in_channels=3, labels=num_classes).to(device)
    elif args.model == 'resnet56':
        model = resnet56(num_classes=num_classes).to(device)  
    elif args.model == 'densenet121':
        model = DenseNet121(num_classes=num_classes).to(device)  
    elif args.model == 'vgg19bn':
        model = vgg19_bn(num_classes=num_classes).to(device)   
    return model