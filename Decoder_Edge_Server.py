import torch
import torch.nn as nn


### 自注意力层
class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        # self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.after_attention_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B, C, W, H)
            returns :
                out : self attention value + input feature
                attention: (B, N, N) (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # (B, N, C//8)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # (B, C//8, N)
        # matrix multiplication
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # (B, N, N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # (B, C, N)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.after_attention_conv(out)
        out = self.gamma * out + x
        return out, attention

class Residual_Transposed_Conv(nn.Module):
    def __init__(self,in_dim,out_dim):
        self.chanel_in = in_dim
        self.out_channels = out_dim
        self.conv_t1 = nn.ConvTranspose2d(self.chanel_in, self.out_channels, 3, stride=2)
        self.conv_t2 = nn.ConvTranspose2d(self.out_channels, self.out_channels, 3, stride=1)
        self.conv_t3 = nn.ConvTranspose2d(self.chanel_in, self.out_channels, 3, stride=1)

    def forward(self,x):
        out_on = self.conv_t1(x)
        out_on = self.conv_t2(out_on)
        out_down = self.conv_t3(x)
        out = out_on + out_down
        return out

class Decoder_edge(nn.Module):
    def __init__(self,in_dim,out_dim):
        self.att1 = Self_Attn(in_dim)
        self.res_transposed1 = Residual_Transposed_Conv(in_dim,64)
        self.att2 = Self_Attn(64)
        self.res_transposed2 = Residual_Transposed_Conv(64,32)
        self.conv_out = nn.Conv2d(in_channels=32,out_channels=out_dim,kernel_size=3,stride=1)

    def forward(self,x):
        out = self.att1(x)
        out = self.res_transposed1(out)
        out = self.att2(out)
        out = self.res_transposed2(out)
        out = self.conv_out(out)
        return out