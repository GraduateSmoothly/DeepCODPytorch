import torch


## 卷积核加正则项
def orthogonal_regularizer(w, device, scale=0.0001):
    """ Defining the Orthogonal regularizer and return the function at last to be used in Conv layer as kernel regularizer"""
    c, _, _, _ = w.shape  # [out_dim, in_dim, w,h]
    w = torch.reshape(w, [c, -1])

    """ Declaring a Identity Tensor of appropriate size"""
    identity = torch.eye(c).to(device)

    """ Regularizer Wt*W - I """
    w_transpose = torch.transpose(w, 0, 1)
    w_mul = torch.matmul(w, w_transpose)
    reg = torch.subtract(w_mul, identity)
    # print(reg.shape)
    """Calculating the Loss Obtained"""
    ortho_loss = torch.sum(reg ** 2) * 0.5
    return scale * ortho_loss


### 计算卷积层的正交正则项
def loss_regulation(model,device):
    extra_loss = 0
    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name:
            # print(name)
            # print(param.shape)
            # print(orthogonal_regularizer(param))
            extra_loss += orthogonal_regularizer(param,device)
    return extra_loss


### 计算一阶拉普拉斯，权重大小压缩
# 幂迭代
a = torch.rand(2, 2, 2, 2)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## 论文作者设置的iteration为1
def spectral_norm(w,device, iteration=5):
    a = w.shape  # [out_dim, in_dim, w,h]
    c= a[0]
    w = torch.reshape(w, [c, -1])
    # print(w)
    u = torch.rand(1, c)
    u_hat = u.to(device)
    for i in range(iteration):
        """
        power iteration
		Usually iteration = 1 will be enough
		"""
        v_ = torch.matmul(u_hat, w)
        # output = x / sqrt(max(sum(x**2), epsilon)),
        v_hat = v_ / torch.norm(v_)  # [1,-1]
        u_ = torch.matmul(v_hat, torch.transpose(w, 0, 1))
        u_hat = u_ / torch.norm(u_)  # [1,c]

    sigma = torch.matmul(torch.matmul(v_hat, torch.transpose(w, 0, 1)), torch.transpose(u_hat, 0, 1))

    w_norm = w / sigma
    w_norm = torch.reshape(w_norm, a)

    return w_norm


# print(spectral_norm(a))
