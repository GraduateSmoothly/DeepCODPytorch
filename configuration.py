import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--centers_initial_range", default=[-1,1], help="numerical range of centers")
parser.add_argument("--num_centers", type=int, default=8, help="number of centers for soft quantization")
parser.add_argument("--regularization_factor_centers", type=int, default=0, help="regularization factor of centers")
parser.add_argument('--compress_ratio', default=0.1,help='the compress ratio of the offloading layer')
parser.add_argument('--sigma', default=1,help='sigma')
parser.add_argument('--batch_size', default=2,help='batch size')
parser.add_argument('--workers', default=1,help='num of workers')
# parser.add_argument('--distributed', type=bool,default=True,help='distributed input')

# parser.add_argument('--datapath', default=r'E:\ILSVRC2012\small_ImageNet_bbox',help='data path')

# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
# parser.add_argument('--epochs', default=90, type=int, metavar='N',
#                     help='number of total epochs to run')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
# parser.add_argument('-b', '--batch-size', default=256, type=int,
#                     metavar='N',
#                     help='mini-batch size (default: 256), this is the total '
#                          'batch size of all GPUs on the current node when '
#                          'using Data Parallel or Distributed Data Parallel')
# parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
#                     metavar='LR', help='initial learning rate', dest='lr')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
# parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
#                     metavar='W', help='weight decay (default: 1e-4)',
#                     dest='weight_decay')
# parser.add_argument('-p', '--print-freq', default=10, type=int,
#                     metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',default=True,
#                     help='use pre-trained model')
# parser.add_argument('--world-size', default=-1, type=int,
#                     help='number of nodes for distributed training')
# parser.add_argument('--rank', default=-1, type=int,
#                     help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                     help='url used to set up distributed training')
# parser.add_argument('--dist-backend', default='nccl', type=str,
#                     help='distributed backend')
# parser.add_argument('--seed', default=None, type=int,
#                     help='seed for initializing training. ')
# parser.add_argument('--gpu', default=None, type=int,
#                     help='GPU id to use.')
# parser.add_argument('--workers', default=1, type=int,
#                     help='GPU id to use.')
# parser.add_argument('--multiprocessing-distributed', action='store_true',
#                     help='Use multi-processing distributed training to launch '
#                          'N processes per node, which has N GPUs. This is the '
#                          'fastest way to use PyTorch for either single node or '
#                          'multi node data parallel training')
## test for validity

# opt = parser.parse_args()
# print(opt.centers_initial_range)