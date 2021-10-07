# Data loading code
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from configuration import *
import torch
args = parser.parse_args()

# 数据存储路径
traindir = os.path.join(r'D:\DAstudy\import_baseline_few-shot\DeepCODPytorch\dataset\ImageNet_minitest', 'train')
valdir = os.path.join(r'D:\DAstudy\import_baseline_few-shot\DeepCODPytorch\dataset\ImageNet_minitest', 'test')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))


# if args.distributed:
#     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
# else:
#     train_sampler = None
train_sampler = None

Imagenet_train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    num_workers=args.workers, pin_memory=True, sampler=train_sampler)


Imagenet_val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)