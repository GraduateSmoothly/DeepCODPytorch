import torch
import torch.nn as nn


def hw_flatten(x):
    x_shape = x.shape
    return x.view(-1, x_shape[1], x_shape[2] * x_shape[3])


### 自注意力层
class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, channel_split, pool_size, pool_stride):
        super(Self_Attn, self).__init__()
        # self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // channel_split, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // channel_split, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1), requires_grad=True)
        # self.gamma = torch.autograd.Variable(torch.ones(1))
        self.after_attention_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.mpool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
        self.softmax = nn.Softmax(dim=-1)  #
        torch.nn.init.kaiming_normal_(self.query_conv.weight)
        torch.nn.init.kaiming_normal_(self.key_conv.weight)
        torch.nn.init.kaiming_normal_(self.value_conv.weight)
        torch.nn.init.kaiming_normal_(self.after_attention_conv.weight)

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B, C, W, H)
            returns :
                out : self attention value + input feature
                attention: (B, N, N) (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = hw_flatten(self.mpool(self.query_conv(x))).permute(0, 2, 1)  # (B, N2, C2)
        proj_key = hw_flatten(self.key_conv(x))  # (B, C2, N)
        # matrix multiplication
        energy = torch.bmm(proj_query, proj_key)  # transpose check

        attention = self.softmax(energy)  # (B, N2, N)

        proj_value = hw_flatten(self.mpool(self.value_conv(x)))  # (B, C2, N2)
        out = torch.bmm(proj_value, attention)  # (B, C2, N)
        out = out.view(m_batchsize, -1, width, height)
        out = self.after_attention_conv(out)
        out = self.gamma * out + x
        # print(self.gamma.item())
        # print('gamma',self.gamma)
        return out


class Residual_Transposed_Conv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Residual_Transposed_Conv, self).__init__()
        self.chanel_in = in_dim
        self.out_channels = out_dim
        self.conv_t1 = nn.ConvTranspose2d(self.chanel_in, self.out_channels, 3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv_t2 = nn.ConvTranspose2d(self.out_channels, self.out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.conv_t3 = nn.ConvTranspose2d(self.chanel_in, self.out_channels, 3, stride=2, padding=1, output_padding=1)
        # self.bn3 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.LeakyReLU()
        torch.nn.init.kaiming_normal_(self.conv_t1.weight)
        torch.nn.init.kaiming_normal_(self.conv_t2.weight)
        torch.nn.init.kaiming_normal_(self.conv_t3.weight)

    def forward(self, x):
        out_on = self.relu(self.bn1(self.conv_t1(x)))
        # print('1',out_on.shape)
        out_on = self.relu(self.bn2(self.conv_t2(out_on)))
        # print('2', out_on.shape)
        out_down = self.conv_t3(x)
        # print('3', out_down.shape)
        out = out_on + out_down
        return out


class Decoder_edge(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Decoder_edge, self).__init__()
        self.att1 = Self_Attn(in_dim, in_dim // out_dim, 4, 4)
        self.res_transposed1 = Residual_Transposed_Conv(in_dim, 64)
        self.att2 = Self_Attn(64, 8, 8, 8)
        self.res_transposed2 = Residual_Transposed_Conv(64, 32)
        self.conv_out = nn.Conv2d(in_channels=32, out_channels=out_dim, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.LeakyReLU()
        self.tan = nn.Tanh()
        # self.conv_final = nn.Conv2d(in_channels=out_dim,out_channels=3,kernel_size=3,stride=1,padding=[1,1])

    def forward(self, x):
        # print(x.shape)
        out = self.att1(x)
        # print(out.shape)
        out = self.res_transposed1(out)
        # print(out.shape)
        out = self.att2(out)
        # print(out.shape)
        out = self.relu(self.bn(self.res_transposed2(out)))
        # print(out.shape)
        out = self.tan(self.conv_out(out))
        # print(out.shape)
        return out
