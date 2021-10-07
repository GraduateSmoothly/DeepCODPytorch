### image recognition
# 调用预训练好的模型
# from torchvision.models import resnet50
from STE_optimizer import *
# from Huffman_encode_and_decode import *
from configuration import *

###################################################################################################
### encoder part
from quantization_part import quantize1d
import torch.nn as nn


class encoder_mobile_part(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(encoder_mobile_part, self).__init__()
        # convolution strides is set to be equal to the kernel size in the paper
        # self.strides = kernel_size
        # self.sigma = sigma
        # self.centers = centers
        # self.data_format = data_format
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, 4)
        self.quantizer = quanti_STE.apply
        torch.nn.init.kaiming_normal_(self.conv_layer.weight)

    def forward(self, x):
        out = self.conv_layer(x)
        # print('conv_shape',out.shape)
        # print(self.centers.grad)
        # out = self.quantizer(out, self.centers,self.sigma, self.data_format)
        # print(self.centers[0]==self.centers[1])
        # print(self.centers.grad)
        return out

        # if self.data_format == 'NHWC':
        #     # NHWC_to_NCHW
        #     NHWC_to_NCHW = lambda x: x.permute(0, 3, 1, 2)
        #     x = NHWC_to_NCHW(x)

        # num_centers = 8
        # x_shape_BCwh = out.shape
        # B = x_shape_BCwh[0]  # B is not necessarily static
        # C = int(out.shape[1])  # C is static
        # x = out.view(B, C, -1).unsqueeze(dim=-1)
        # phi_soft = F.softmax(-self.sigma * torch.abs(x - self.centers), dim=-1)
        # phi_hard = F.softmax(-1e7 * torch.abs(x - self.centers)**2, dim=-1)
        # symbols_hard = torch.argmax(phi_hard, axis=-1)
        # symbols_hard_temp = symbols_hard.view(-1)
        # phi_hard = torch.nn.functional.one_hot(symbols_hard_temp, num_classes=num_centers).view(x_shape_BCwh[0],x_shape_BCwh[1],-1,phi_hard.shape[-1])
        # hardout = (phi_hard * self.centers).sum(dim=-1).view(x_shape_BCwh[0], x_shape_BCwh[1], x_shape_BCwh[2],
        #                                                      x_shape_BCwh[3])
        # softout = (phi_soft * self.centers).sum(dim=-1).view(x_shape_BCwh[0], x_shape_BCwh[1], x_shape_BCwh[2],
        #                                                      x_shape_BCwh[3])
        # softout = nn.Parameter(softout, requires_grad=True)
        # # out2 = self.gamma - 1
        # out = torch.add(hardout, -softout).detach() + softout
        #
        # # out3 = out2 * self.gamma
        # # print(self.gamma.grad)
        # # print(out.shape)
        # # softout, hardout, symbols_hard = quantize1d(out, self.centers, self.sigma, self.data_format)
        # # hardout = nn.Parameter(hardout)
        # ### 这里返回softout是为了让反向传播的时候能利用这个计算梯度
        # # out = torch.add(hardout, -softout).detach() + softout
        # print(self.centers.grad)

        # return out
