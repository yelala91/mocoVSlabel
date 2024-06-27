
import torch.utils
import torch.utils.data
import utils
import numpy as np

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18
from torch.optim.lr_scheduler import CosineAnnealingLR

import train
from model.MoCo import MoCo
from model.myResNet import myResNet
import os

def main():
    args = utils.make_arg().parse_args()
    
    train_transform = transforms.Compose(
        [transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
        ])
    
    test_transform = transforms.Compose(
        [transforms.Resize((224, 224)),  
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]) 
        ])
    
    # trainset = utils.CutMix_TinyImageNet(args.dataset, num_class=200, transform=train_transform, is_super=(args.train_style=='Supervised'))
    # train_loader = torch.utils.data.DataLoader(
    #     trainset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.workers,
    #     drop_last=True
    # )
    
    
    if args.mode == 'train':
        if args.model == 'moco':
            trainset = utils.TinyImageNet(args.dataset, num_class=200, transform=train_transform)
            model = MoCo(
                resnet18, 
                dim=args.moco_dim,
                K=args.moco_k,
                m=args.moco_m,
                T=args.moco_t)
            weight_name = 'moco.pth'
            
            train_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                drop_last=True
            )
            
            loaders = train_loader
        elif args.model == 'resnet':
            trainset = utils.CIFAR(args.dataset, load_type='train', transform=train_transform)
            validset = utils.CIFAR(args.dataset, load_type='valid', transform=test_transform)
            
            train_loader = torch.utils.data.DataLoader(
                trainset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=args.workers,
                drop_last=True)
        
            valid_loader = torch.utils.data.DataLoader(
                validset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                drop_last=True)
            
            model = myResNet(100)
            weight_name = 'resnet.pth'
        
            loaders = (train_loader, valid_loader)

        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=args.lr, 
            momentum=args.momentum,
            weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss().to(args.device)
        scheduler = CosineAnnealingLR(
            optimizer, 
            int(args.epochs), 
            eta_min=0.003)
            
        train.train(model, loaders, optimizer, criterion, scheduler, args)
        torch.save(model.state_dict(), os.path.join(args.out_dir, weight_name))

    elif args.mode == 'LCPtrain':
        
        trainset = utils.CIFAR(args.dataset, load_type='train', transform=train_transform)
        validset = utils.CIFAR(args.dataset, load_type='valid', transform=test_transform)
            
        train_loader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.workers,
            drop_last=True)
    
        valid_loader = torch.utils.data.DataLoader(
            validset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True)
        
        loaders = (train_loader, valid_loader)
        model = myResNet(num_classes=100)
        # weight_path = args.weight if args.model == 'moco' else './weights/resnet18-ImageNet.pth'
        weights = torch.load(args.weight)
        if args.model == 'moco':
            weights = utils.get_resnet_weight(weights)
        
        del weights['fc.weight']
        del weights['fc.bias']
        model.load_state_dict(weights, strict=False)
        
        optimizer = torch.optim.SGD(
            model.fc.parameters(), 
            lr=args.lr, 
            momentum=args.momentum,
            weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss().to(args.device)
        scheduler = CosineAnnealingLR(
            optimizer, 
            args.epochs, 
            eta_min=0)
        
        weight_name = 'LCP_moco_model.pth' if args.model == 'moco' else 'LCP_ImageNet_model.pth'
            
        train.train(model, loaders, optimizer, criterion, scheduler, args)
        torch.save(model.state_dict(), os.path.join(args.out_dir, weight_name))
        
    elif args.mode == 'ft':
        ft_rate = 0.1
        trainset = utils.CIFAR(args.dataset, load_type='train', transform=train_transform)
        validset = utils.CIFAR(args.dataset, load_type='valid', transform=test_transform)
            
        train_loader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.workers,
            drop_last=True)
    
        valid_loader = torch.utils.data.DataLoader(
            validset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True)
        
        loaders = (train_loader, valid_loader)
        model = myResNet(num_classes=100)
        # weight_path = args.weight if args.model == 'moco' else './weights/resnet18-ImageNet.pth'
        
        if args.weight != None:
            weights = torch.load(args.weight)
        
        if args.model == 'moco':
            weights = utils.get_resnet_weight(weights)
        
        if args.weight != None:
            del weights['fc.weight']
            del weights['fc.bias']
            model.load_state_dict(weights, strict=False)
    
        optimizer = torch.optim.SGD(
            model.fc.parameters(), 
            lr=args.lr, 
            momentum=args.momentum,
            weight_decay=args.weight_decay)
        optimizer.add_param_group({
            'params': list(model.parameters())[:-2], 
            'lr': ft_rate*args.lr, 
            'momentum':args.momentum,
            'weight_decay':args.weight_decay})
        
        criterion = nn.CrossEntropyLoss().to(args.device)
        scheduler = CosineAnnealingLR(
            optimizer, 
            args.epochs, 
            eta_min=0)
        if args.weight is None:
            weight_name = 'zero_model.pth'
        else:
            weight_name = 'ft_moco_model.pth' if args.model == 'moco' else 'ft_ImageNet_model.pth'
        
        train.train(model, loaders, optimizer, criterion, scheduler, args)
        torch.save(model.state_dict(), os.path.join(args.out_dir, weight_name))
        
    elif args.mode == 'test':
        testset = utils.CIFAR(args.dataset, load_type='test', transform=test_transform)
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True
        )
        model = myResNet(100); 
        weights = torch.load(args.weight)
        model.load_state_dict(weights)
        
        acc = train.test(model, test_loader, args)
        print(f'Acc is {acc*100: .2f}%')
            
if __name__ == '__main__':
    main()