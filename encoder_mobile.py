### image recognition
# 调用预训练好的模型
from torchvision.models import resnet50
from STE_optimizer import *
from Huffman_encode_and_decode import *
resnet_pretrained = resnet50(pretrained=True, progress=True)

###################################################################################################


### encoder part
import torch
import torch.nn as nn
from configuration import *
class encoder_mobile(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,centers, data_format):
        # convolution strides is set to be equal to the kernel size in the paper
        self.strides = kernel_size
        self.centers = centers
        self.data_format = data_format
        self.conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=self.strides, padding=0,
                                          groups=1, bias=True)
        self.quantizer = quanti_STE.apply

    def forward(self, x,data_format):
        out = self.conv_layer(x)
        out = self.quantizer(out, self.centers,1,self.data_format)
        ### 下面接huffman编码
        out_text = out.view(out.shape[0]*out.shape[1],-1)
        out,_ = huffman_encode_decode()
        return out
        ### softout用于梯度更新，hardout用于前向传播，symbols_hard直接代表对应的center的下标
        # softout, hardout, symbols_hard = _quantize1d(out1, centers, sigma, data_format)
        # return softout, hardout, symbols_hard

sigma = 1  ## for smooth
data_format = 'NCHW'
config = parser.parse_args()
### 初始化centers
centers = create_centers_variable(config)

