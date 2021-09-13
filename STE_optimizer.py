import torch
from quantization_part import *
from quantization_part import _quantize1d

class quanti_STE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, centers, sigma, data_format):
        softout, hardout, symbols_hard = _quantize1d(input, centers, sigma, data_format)
        ### 这里返回softout是为了让反向传播的时候能利用这个计算梯度
        no_grad_out_part1 = torch.add(hardout,-softout)
        return no_grad_out_part1.detach()+softout

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)

# import torch
# from LBSign import LBSign

if __name__ == '__main__':

    sign = quanti_STE.apply
    params = torch.randn(4, requires_grad = True)
    output = sign(params)
    loss = output.mean()
    print(loss)
    loss.backward()



# #### 验证detach函数
# import torch
# x = torch.tensor(([1.0]),requires_grad=True)
# y = 2*x
# z = (y+x)
# # w= z-x
#
# # detach it, so the gradient w.r.t `p` does not effect `z`!
# # p = z.detach()
# p=z.detach()
# # q = torch.tensor(([2.0]), requires_grad=True)
# pq = p-x
# pq.backward(retain_graph=True)
# # w.backward()
# print(x.grad)