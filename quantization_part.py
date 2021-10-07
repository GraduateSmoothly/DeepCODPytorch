import torch
import torch.nn as nn
import torch.nn.functional as F

_HARD_SIGMA = 1e7

################## 量化部分 ####################
### give the number of centers, create uniform random centers within a limited interval, dtype = float32
def create_centers_variable(config):  # (C, L) or (L,)
    assert config.num_centers is not None
    # obtain same results when run the .py
    torch.manual_seed(666)
    minval, maxval = map(int, config.centers_initial_range)
    centers = torch.rand(config.num_centers, dtype=torch.float32) * (maxval - minval) - maxval
    return centers


def create_centers_regularization_term(config, centers):
    # Add centers regularization
    if config.regularization_factor_centers != 0:
        # other tensor to float tensor
        reg = config.regularization_factor_centers.float()
        temp = reg * torch.sum(centers ** 2) * 0.5
        centers_reg = temp.clone(temp)
        ## 少一句，上式是损失函数的构成之一,下面需要加一句把centers_reg加入到损失函数，这里直接返回该值，之后加入损失函数
        return centers_reg



### 得到软量化，硬量化，和硬量化得到的最大下标，其中硬量化是前向传播中使用，软量化仅在反向传播中使用
def quantize1d(x, centers,sigma, data_format):
    """
    :return: (softout, hardout, symbols_vol)
        each of same shape as x, softout, hardout will be float32, symbols_vol will be int64
    """

    assert x.type() == 'torch.FloatTensor'
    assert centers.type() == 'torch.FloatTensor'
    assert len(x.shape) == 4, 'x should be NCHW or NHWC, got {}'.format(x.shape())
    # assert len(centers.shape) == 4, 'centers should be (L,), got {}'.format(centers.shape())

    if data_format == 'NHWC':
        # NHWC_to_NCHW
        NHWC_to_NCHW = lambda x: x.permute(0, 3, 1, 2)
        x_t = NHWC_to_NCHW(x)
        softout, hardout, symbols_hard = quantize1d(x_t, centers, sigma, data_format='NCHW')
        NCHW_to_NHWC = lambda x: x.permute(0, 3, 1, 2)
        return tuple(map(NCHW_to_NHWC, (softout, hardout, symbols_hard)))

    # Note: from here on down, x is NCHW ---

    # count centers
    num_centers = centers.shape[-1]

    # with tf.name_scope('reshape_BCm1'):
    # reshape (B, C, w, h) to (B, C, m=w*h)
    x_shape_BCwh = x.shape
    B = x_shape_BCwh[0]  # B is not necessarily static
    C = int(x.shape[1])  # C is static
    x = x.view(B, C, -1)

    # make x into (B, C, m, 1)
    x = x.unsqueeze(dim=-1)

    # with tf.name_scope('dist'):
    # dist is (B, C, m, L), contains | x_i - c_j | ^ 2
    ### dists是用来算硬量化的，即对应的center的, 这里x和centers的维度不一样, eg. (3,1)的矩阵减去(1,5)的矩阵结果为(3,5)的矩阵
    dist = torch.abs(x - centers) ** 2
    ### dist_Euclidean是用来算软量化的，作者使用该距离的softmax对centers进行学习
    dist_Euclidean = torch.abs(x - centers)

    # with tf.name_scope('phi_soft'):
    # (B, C, m, L)
    phi_soft = F.softmax(-sigma * dist_Euclidean, dim=-1)

    # with tf.name_scope('phi_hard'):
    # (B, C, m, L) probably not necessary due to the argmax!
    phi_hard = F.softmax(-_HARD_SIGMA * dist, dim=-1)
    ### symbols_hard 代表的是第几个center, (B,C,m)
    symbols_hard = torch.argmax(phi_hard, axis=-1)
    # phi_hard = tf.one_hot(symbols_hard, depth=num_centers, axis=-1, dtype=tf.float32)
    ### 这里的phi_hard 代表的是将第几个center进行one-hot编码，表明其所在位置
    symbols_hard_temp = symbols_hard.view(-1)
    phi_hard = torch.nn.functional.one_hot(symbols_hard_temp,num_classes = num_centers).view(x_shape_BCwh[0],x_shape_BCwh[1],-1,phi_hard.shape[-1])
    # with tf.name_scope('softout'):
    softout = phi_times_centers(phi_soft, centers)

    # with tf.name_scope('hardout'):
    ### hardout 代表之前的one-hot与centers向量相乘，结果是仅保留其对应的center
    hardout = phi_times_centers(phi_hard, centers)

    # (B, C, m) to (B, C, w, h)
    reshape_to_BCwh = lambda x: x.view(x_shape_BCwh[0],x_shape_BCwh[1],x_shape_BCwh[2],x_shape_BCwh[3])
    # def reshape_to_BCwh(t_):
    #     # with tf.name_scope('reshape_BCwh'):
    #     return t_.view_as(x_shape_BCwh)
    softout, hardout, symbols_hard = tuple(map(reshape_to_BCwh, (softout, hardout, symbols_hard)))
    softout = nn.Parameter(softout)
    hardout = nn.Parameter(hardout)
    # symbols_hard = nn.Parameter(symbols_hard)
    # softout.retain_grad()
    # print(softout.requires_grad)
    # print(centers.grad_fn)
    return softout, hardout, symbols_hard


### (B, C, m, L) to (B, C, m)
def phi_times_centers(phi, centers):
    matmul_innerproduct = phi * centers  # (B, C, m, L)
    # print(centers.grad)
    return matmul_innerproduct.sum(dim=-1)

#[ [ [[],   []],   [] ],   []  ]

# from configuration import *
# x = torch.nn.Parameter(torch.rand(2,3,4,8))
# config = parser.parse_args()
# # sigma
# sigma = config.sigma  ## for smooth
# centers = torch.nn.Parameter(create_centers_variable(config))
# # y = out*centers
# # loss = y.sum()
# #
# num_centers = centers.shape[-1]
# x_shape_BCwh = x.shape
# B = x_shape_BCwh[0]  # B is not necessarily static
# C = int(x.shape[1])  # C is static
# x1 = x.view(B, C, -1)
# x1 = x1.unsqueeze(dim=-1)
# dist = torch.abs(x1 - centers) ** 2
# dist_Euclidean = torch.abs(x1 - centers)
# phi_soft = F.softmax(-sigma * dist_Euclidean, dim=-1)
# phi_hard = F.softmax(-1e7 * dist, dim=-1)
# symbols_hard = torch.argmax(phi_hard, axis=-1)
# symbols_hard_temp = symbols_hard.view(-1)
# phi_hard = torch.nn.functional.one_hot(symbols_hard_temp, num_classes=num_centers)
# hardout = (phi_hard * centers).sum(dim=-1).view(x_shape_BCwh[0], x_shape_BCwh[1], x_shape_BCwh[2],
#                                                      x_shape_BCwh[3])
# softout = (phi_soft * centers).sum(dim=-1).view(x_shape_BCwh[0], x_shape_BCwh[1], x_shape_BCwh[2],
#                                                      x_shape_BCwh[3])
# #
# # softout, hardout, symbols_hard,centers = quantize1d(x, centers, sigma, 'NCHW')
# # softout.retain_grad()
#
# loss = softout.sum()
# loss.backward()
# #
# # # print(softout.grad)
# print(x.grad == None)
# print(x.is_leaf)
# print(x.requires_grad)
# print(centers.grad)
# print(centers.requires_grad)
# print(centers.is_leaf)


def center_quatization(x,centers,sigma, data_format,device):
    if data_format == 'NHWC':
        # NHWC_to_NCHW
        NHWC_to_NCHW = lambda x: x.permute(0, 3, 1, 2)
        x = NHWC_to_NCHW(x)
    num_centers = centers.shape[-1]
    x_shape_BCwh = x.shape
    B = x_shape_BCwh[0]  # B is not necessarily static
    C = int(x.shape[1])  # C is static
    x = x.view(B, C, -1)
    x = x.unsqueeze(dim=-1).to(device)
    centers = centers.to(device)
    # dist = torch.abs(x - centers) ** 2
    dist_Euclidean = torch.abs(x - centers)
    phi_soft = F.softmax(-sigma * dist_Euclidean, dim=-1)
    # phi_hard = F.softmax(-1e7 * dist, dim=-1)
    # symbols_hard = torch.argmax(phi_hard, axis=-1)
    symbols_hard = torch.argmax(phi_soft, axis=-1)
    # symbols_hard_temp = symbols_hard.view(-1)

    phi_hard = torch.nn.functional.one_hot(symbols_hard.view(-1), num_classes=num_centers)
    hardout = (phi_hard * centers).sum(dim=-1).view(x_shape_BCwh[0], x_shape_BCwh[1], x_shape_BCwh[2],
                                                    x_shape_BCwh[3])
    softout = (phi_soft * centers).sum(dim=-1).view(x_shape_BCwh[0], x_shape_BCwh[1], x_shape_BCwh[2],
                                                    x_shape_BCwh[3])
    # softout = nn.Parameter(softout)
    out = torch.add(hardout, -softout).detach() + softout
    return symbols_hard


class q(nn.Module):
    def __init__(self,data_format,centers):
        super(q, self).__init__()
        self.centers = centers
        self.sigma = 1
        self.data_format = data_format

    def forward(self,x):
        if self.data_format == 'NHWC':
            # NHWC_to_NCHW
            NHWC_to_NCHW = lambda x: x.permute(0, 3, 1, 2)
            x = NHWC_to_NCHW(x)
        num_centers = self.centers.shape[-1]
        x_shape_BCwh = x.shape
        B = x_shape_BCwh[0]  # B is not necessarily static
        C = int(x.shape[1])  # C is static
        x = x.view(B, C, -1)
        x = x.unsqueeze(dim=-1)
        dist = torch.abs(x - self.centers) ** 2
        dist_Euclidean = torch.abs(x - self.centers)
        phi_soft = F.softmax(-self.sigma * dist_Euclidean, dim=-1)
        phi_hard = F.softmax(-1e7 * dist, dim=-1)
        symbols_hard = torch.argmax(phi_hard, axis=-1)
        symbols_hard_temp = symbols_hard.view(-1)
        phi_hard = torch.nn.functional.one_hot(symbols_hard_temp, num_classes=num_centers)
        hardout = (phi_hard * self.centers).sum(dim=-1).view(x_shape_BCwh[0], x_shape_BCwh[1], x_shape_BCwh[2],
                                                             x_shape_BCwh[3])
        softout = (phi_soft * self.centers).sum(dim=-1).view(x_shape_BCwh[0], x_shape_BCwh[1], x_shape_BCwh[2],
                                                             x_shape_BCwh[3])
        # softout = self.te * softout
        out = torch.add(hardout, -softout).detach() + torch.nn.Parameter(softout, requires_grad=True)
        print('centers',self.centers.grad)
        # print(self.te)
        # print('cpm',self.conv_layer.weight.grad)
        return out