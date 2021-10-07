import torch
from quantization_part import *
from quantization_part import quantize1d

class quanti_STE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, centers, sigma, data_format):
        softout, hardout, symbols_hard = quantize1d(input, centers, sigma, data_format)
        # num_centers = centers.shape[-1]
        # x_shape_BCwh = input.shape
        # B = x_shape_BCwh[0]  # B is not necessarily static
        # C = int(input.shape[1])  # C is static
        # input1 = input.view(B, C, -1)
        # input1 = input1.unsqueeze(dim=-1)
        # dist = torch.abs(input1 - centers) ** 2
        # dist_Euclidean = torch.abs(input1 - centers)
        # phi_soft = F.softmax(-sigma * dist_Euclidean, dim=-1)
        # phi_hard = F.softmax(-1e7 * dist, dim=-1)
        # symbols_hard = torch.argmax(phi_hard, axis=-1)
        # symbols_hard_temp = symbols_hard.view(-1)
        # phi_hard = torch.nn.functional.one_hot(symbols_hard_temp, num_classes=num_centers)
        # hardout = (phi_hard * centers).sum(dim=-1).view(x_shape_BCwh[0], x_shape_BCwh[1], x_shape_BCwh[2],
        #                                                      x_shape_BCwh[3])
        # softout = (phi_soft * centers).sum(dim=-1).view(x_shape_BCwh[0], x_shape_BCwh[1], x_shape_BCwh[2],
        #                                                      x_shape_BCwh[3])
        # ### 这里返回softout是为了让反向传播的时候能利用这个计算梯度
        out = torch.add(hardout,-softout).detach()+softout
        # ctx.save_for_backward(input, centers, softout)
        # softout.retain_grad()
        # print('centers', centers.grad)
        # print('sdadsadsa',softout.grad)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        print(grad_output.shape)
        return grad_output.clamp_(-1, 1),None,None,None

# import torch
# from LBSign import LBSign

# if __name__ == '__main__':
#
#     sign = quanti_STE.apply
#     params = torch.randn(4, requires_grad = True)
#     output = sign(params)
#     loss = output.mean()
#     print(loss)
#     loss.backward()



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