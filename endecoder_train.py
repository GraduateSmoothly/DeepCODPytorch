import time
from earlystopping import *
from torchvision.models import resnet50
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
from encoder_mobile import *
from Decoder_Edge_Server import *
from conv_regulation import *
# from configuration import *
# from torchvision import datasets, transforms
import os
from dataload.ImageNet_processsing import *
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
### 规定运行设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
### 训练数据导入
### 定义预处理函数

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# ### 加载数据集,训练数据集和测试数据集都要加载，第一个参数是存数据的路径，如果路径是root，则数据在root/MNIST里，若没有则download=True
# ### 如果报错一般是pytorch版本问题，只需要把以前的数据集删掉，然后download=True
# minist_t = datasets.MNIST(r'D:\DAstudy\import_baseline_few-shot\pytorch_pruning_quatization_test', download=False, \
#                           train=True, transform=transform)
# minist_e = datasets.MNIST(r'D:\DAstudy\import_baseline_few-shot\pytorch_pruning_quatization_test', \
#                           train=False, transform=transform)
#
# ### 给出数据集和batch_size
# train_dataloader = torch.utils.data.DataLoader(minist_t, batch_size=32, shuffle=True)
# test_loader = torch.utils.data.DataLoader(minist_e, batch_size=1000)



### 实例化分类网络，这里采用预训练好的resnet50
Resnet50 = resnet50(pretrained=True, progress=True)
### 设置编码器基本参数

# 设置数据格式
data_format = 'NCHW'
# # 导入参数
config = parser.parse_args()
# sigma
sigma = config.sigma  ## for smooth

# 初始化核的大小
kernel_size = 4
# 给定编码器的路径
encoder_model_path = ''
# 给出编码器输入维度,即前半段网络输出的维度
in_channels = 3
# 设立编码器输出维度，本应该是fastdeepiot来确定的
out_channels = max(int(3 * config.compress_ratio * 4 * 4), 1)
### 实例化编码器
centers = nn.Parameter(create_centers_variable(config))
# regu_param = nn.Parameter(torch.FloatTensor([0.5]),requires_grad=True)
# [B,C,W,H] 已量化压缩
# Encoder_on_device = encoder_mobile_part(in_channels, out_channels, kernel_size).to(device)

### 设置解码器基本参数
in_dim = out_channels
out_dim = 3
### 实例化解码器
# 最终返回的还是通道为3的图像
# Decoder_on_cloud = Decoder_edge(in_dim, out_dim).to(device)

encoder_path = r'D:\DAstudy\import_baseline_few-shot\DeepCODPytorch\endecoder_save\best_encoder.pt'
decoder_path = r'D:\DAstudy\import_baseline_few-shot\DeepCODPytorch\endecoder_save\best_decoder.pt'
Encoder_on_device = encoder_mobile_part(in_channels, out_channels, kernel_size)
Decoder_on_cloud = Decoder_edge(in_dim, out_dim)
Encoder_on_device.load_state_dict(torch.load(encoder_path))
Encoder_on_device = Encoder_on_device.to(device)
Decoder_on_cloud.load_state_dict(torch.load(decoder_path))
Decoder_on_cloud = Decoder_on_cloud.to(device)

### 定义优化器，损失函数等
# optimizer_e = optim.SGD(Encoder_on_device.parameters(), lr=0.01)
# optimizer_d = optim.SGD(Decoder_on_cloud.parameters(), lr=0.01)
optimizer = optim.SGD([
    {'params': Encoder_on_device.parameters()},
    {'params': Decoder_on_cloud.parameters()},
    {'params': centers}], lr=0.0001,momentum=0.9,weight_decay=1e-4)
criterion = nn.MSELoss()
# 每隔step_size个epoch，更新一次学习率
# scheduler_e = StepLR(optimizer_e, step_size=2, gamma=0.7)
# scheduler_d = StepLR(optimizer_e, step_size=2, gamma=0.7)
# scheduler = StepLR(optimizer, step_size=2, gamma=0.7)
scheduler = CosineAnnealingLR(optimizer,T_max=5)

# 训练时，对重构的图像和原图像做MSELoss
# 训练收敛后，知识蒸馏时，重构图像和原图像都经过网络，对经过残差网络每一个block的结果做MSELoss，然后求和
# 在知识蒸馏时，需要固定残差网络的参数，冻结其参数，只更新编码器和解码器


## 权重按最大特征值归一化
def weight_spectral_norm(model_name='model'):
    for name, param in eval("%s.named_parameters()" % model_name):
        # if 'centers' in name:
        #     print('center',param)
        if 'weight' in name and 'conv' in name:
            # print(name)
            b = spectral_norm(param,device)
            b = torch.nn.Parameter(b)
            exec("%s.%s=b" % (model_name, name))


grads = {}  # 存储节点名称与节点的grad


def save_grad(name):
    def hook(grad):
        grads[name] = grad

    return hook

def loss_regu(loss,loss1,loss2,regu):
    regu = regu.to(device)
    return loss+abs(regu)*(loss1+loss2)
### train函数
### 第三步，定义训练函数,这里的train函数只训练一个epoch
def train(epoch, train_dataloader, centers):
    centers_path = r'D:\DAstudy\import_baseline_few-shot\DeepCODPytorch\endecoder_save\centers.txt'
    # regu_param_path = r'D:\DAstudy\import_baseline_few-shot\DeepCODPytorch\endecoder_save\regu_param.txt'
    if os.path.exists(centers_path):
        with open(centers_path, 'r') as f1:
            centers = torch.Tensor(eval(f1.read()))
        f1.close()
    Encoder_on_device.train()
    Decoder_on_cloud.train()
    ## 从dataloader中加载数据，这里的数据是一个batch一个batch的加载的
    for batch_idx, (data, label) in enumerate(train_dataloader):
        ## 将数据传递给设备
        data, label = data.to(device), label.to(device)
        # imshow(data[0])
        ## 将上一个batch计算得到的loss清零，防止影响该batch梯度更新
        # optimizer_e.zero_grad()
        # optimizer_d.zero_grad()
        optimizer.zero_grad()

        ## 将数据导入网络模型
        encode_out = Encoder_on_device(data)
        # encode_out.register_hook(save_grad('encoder_out'))
        # print('en',encode_out.grad)
        # centers = centers
        # print('en', centers)
        encode_out = center_quatization(encode_out, centers, sigma, data_format,device)
        decode_out = Decoder_on_cloud(encode_out)
        # imshow(decode_out[0])
        # print('before',Encoder_on_device.conv_layer.weight[0][0])
        # ## 调用幂迭代改变网络权重
        weight_spectral_norm('Encoder_on_device')
        weight_spectral_norm('Decoder_on_cloud')

        # print('after',Encoder_on_device.conv_layer.weight[0][0])
        ## 将网络输出和label比较，计算损失函数
        loss = criterion(data, decode_out)
        ## 加上卷积核正交正则惩罚项
        loss += 0.0001*(loss_regulation(Encoder_on_device,device) + loss_regulation(Decoder_on_cloud,device))
        # loss = loss_regu(loss, loss_regulation(Encoder_on_device,device), loss_regulation(Decoder_on_cloud,device),regu_param)
        # print(regu_param)
        # loss.retain_grad()
        # loss = abs(loss-0.3) +0.3
        ## 损失函数反向传播
        loss.backward()

        # print('enc',grads['centers'])
        # for name,param in Encoder_on_device.named_parameters():
        #     if 'centers' in name:
        #     # print(name)
        #         print(param.grad)
        optimizer.step()
        # print(Encoder_on_device.centers.grad)
        # print(centers[0])

        # print(param[0][0])
        # for name,param in Decoder_on_cloud.named_parameters():
        #     print(name)

        ## 输出实时结果,每隔100个batch输出一次
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_dataloader.dataset),
                       100. * batch_idx / len(train_dataloader), loss.item()))
            # evaluate on validation set
    return centers
    # if batch_idx % 300 ==0:
    #     test_loss = test(test_loader)
    #
    #     # remember best acc@1 and save checkpoint
    #     if test_loss < min_loss:
    #         state_dict_encoder = Encoder_on_device.state_dict()
    #         state_dict_decoder = Decoder_on_cloud.state_dict()
    #         torch.save(state_dict_encoder, encoder_path)
    #         torch.save(state_dict_decoder, decoder_path)
    #         min_loss = test_loss


### test函数
### 第四步，定义测试函数
def test(test_loader, centers):
    # model.eval()

    centers_path = r'D:\DAstudy\import_baseline_few-shot\DeepCODPytorch\endecoder_save\centers.txt'
    # regu_param_path = r'D:\DAstudy\import_baseline_few-shot\DeepCODPytorch\endecoder_save\regu_param.txt'
    if os.path.exists(centers_path):
        with open(centers_path, 'r') as f1:
            centers = torch.Tensor(eval(f1.read()))
        f1.close()

    Encoder_on_device.eval()
    Decoder_on_cloud.eval()
    ## 记录loss
    test_loss = 0
    # print(len(test_loader))
    ## 关闭梯度更新
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            ## 将数据传递给设备
            # print(data[0])
            data, label = data.to(device), label.to(device)
            ## 将数据导入网络模型
            centers = centers.to(device)
            encode_out = Encoder_on_device(data)
            encode_out = center_quatization(encode_out, centers, sigma, data_format,device)
            decode_out = Decoder_on_cloud(encode_out)
            # # ## 调用幂迭代改变网络权重
            weight_spectral_norm('Encoder_on_device')
            weight_spectral_norm('Decoder_on_cloud')
            # print(decode_out.shape)
            # imshow(data[0])
            if batch_idx%10 == 0:
                imshow(decode_out[0])
            ## 将网络输出和label比较，计算损失函数
            test_loss += criterion(data, decode_out)
            # print('en', centers)
        # test_loss = test_loss / len(test_loader.dataset)

        # if test_loss < min_loss:
        #     torch.save(Encoder_on_device, encoder_path)
        #     torch.save(Decoder_on_cloud, decoder_path)
        #     with open(centers_path, 'w') as f:
        #         f.write(str(centers.tolist()))
        #     f.close()
        #     with open(regu_param_path, 'w') as f3:
        #         f3.write(str(regu_param.tolist()))
        #     f3.close()
            # min_loss = test_loss
        ## 输出结果
        print('\nTest loss: {:.4f}\t'.format(test_loss/len(test_loader)))
    return test_loss/len(test_loader)


def knowledge_distill(model_name, train_dataloader,epoch):
    # encoder_path = r'D:\DAstudy\import_baseline_few-shot\DeepCODPytorch\endecoder_save\best_encoder.pth'
    # decoder_path = r'D:\DAstudy\import_baseline_few-shot\DeepCODPytorch\endecoder_save\best_decoder.pth'
    # encoder_t = encoder_mobile_part(in_channels, out_channels, kernel_size, data_format)
    # decoder_t = Decoder_edge(in_dim, out_dim)
    # encoder_t.load_state_dict(torch.load(encoder_path))
    # decoder_t.load_state_dict(torch.load(decoder_path))
    Encoder_on_device.train()
    Decoder_on_cloud.train()
    total_loss = 0
    for batch_idx, (data, label) in enumerate(train_dataloader):
        loss = 0
        ## 将数据传递给设备
        data, label = data.to(device), label.to(device)

        if model_name == 'resnet50':
            model = resnet50(pretrained=True, progress=True)
            model = model.to(device)
        # 还有yolov5和其他的一些模型
        for param in model.parameters():
            param.requires_grad = False
        ## 将上一个batch计算得到的loss清零，防止影响该batch梯度更新
        optimizer.zero_grad()

        ## 将数据导入网络模型
        encode_out = Encoder_on_device(data)
        decode_out = Decoder_on_cloud(encode_out)
        # imshow(decode_out[0])
        ## 调用幂迭代改变网络权重
        weight_spectral_norm('Encoder_on_device')
        weight_spectral_norm('Decoder_on_cloud')

        ## 读取网络中间层特征
        features = []

        def hook(module, input, output):
            # module: model.conv2
            # input :in forward function  [#2]
            # output:is  [#3 self.conv2(out)]
            features.append(output.cpu().detach())

        hook1 = model.layer1.register_forward_hook(hook)
        hook2 = model.layer2.register_forward_hook(hook)
        hook3 = model.layer3.register_forward_hook(hook)
        hook4 = model.layer4.register_forward_hook(hook)
        out1 = model(data)
        out2 = model(decode_out)
        # features中前四个是存的data的中间特征，后四个存的是重构图像的中间特征

        # print(features[0].shape)
        # print(features[1].shape)
        hook1.remove()
        hook2.remove()
        hook3.remove()
        hook4.remove()

        for i in range(4):
            # print(loss)
            loss += criterion(features[i], features[i + 4])

        ## 加上卷积核正交正则惩罚项
        # loss += 0.0001*(loss_regulation(Encoder_on_device) + loss_regulation(Decoder_on_cloud))
        ## 损失函数反向传播
        # print(loss)
        loss = loss + criterion(out1,out2) - criterion(out1,out2)
        # loss-=criterion(out1,out2)
        loss.backward()
        # print(loss)
        ##　根据反向传播更新网络参数
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_dataloader.dataset),
                       100. * batch_idx / len(train_dataloader), loss.item()))
        total_loss +=loss
    return total_loss/len(train_dataloader)
        # evaluate on validation set
        # test_loss = test(test_loader)
        # encoder_distill_path = r'D:\DAstudy\import_baseline_few-shot\DeepCODPytorch\endecoder_save\best_encoder_distill.pth'
        # decoder_distill_path = r'D:\DAstudy\import_baseline_few-shot\DeepCODPytorch\endecoder_save\best_decoder_distill.pth'
        # # remember best acc@1 and save checkpoint
        # if test_loss < min_loss:
        #     state_dict_encoder = Encoder_on_device.state_dict()
        #     state_dict_decoder = Decoder_on_cloud.state_dict()
        #     torch.save(state_dict_encoder, encoder_distill_path)
        #     torch.save(state_dict_decoder, decoder_distill_path)
        #     min_loss = test_loss


from PIL import Image
import matplotlib.pyplot as plt
import numpy


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    # image = numpy.array(image)
    # print(image)
    # image *= 255  # 变换为0-255的灰度值
    # im = Image.fromarray(image)
    # image = im.convert('L')

    unloader = transforms.ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.draw()
    plt.pause(1)  # pause a bit so that plots are updated
    plt.close()
    # time.sleep(0.01)


if __name__ == '__main__':
    ### 定义训练的epoch数量以及预训练模型的保存路径
    # print(len(Imagenet_val_loader))
    num_epochs = 20
    # min_loss = 50
    # early_stopping = EarlyStopping(best_score=0.3650)
    early_stopping = EarlyStopping(best_score=0.3641)
    encoder_path = r'D:\DAstudy\import_baseline_few-shot\DeepCODPytorch\endecoder_save\best_encoder.pt'
    decoder_path = r'D:\DAstudy\import_baseline_few-shot\DeepCODPytorch\endecoder_save\best_decoder.pt'
    for i in range(num_epochs):
        centers = train(i, Imagenet_train_loader, centers)
        tem_loss = test(Imagenet_val_loader, centers)

        # early stopping
        early_stopping(tem_loss, Encoder_on_device,Decoder_on_cloud,encoder_path,decoder_path, centers)
        if early_stopping.early_stop:
            print("Early stopping.")
            break
        scheduler.step()

    print(test(Imagenet_val_loader, centers))
    # print('*'*20,'distill','*'*20)
    # for i in range(num_epochs):
    #     know_loss = knowledge_distill('resnet50',Imagenet_train_loader,i)
    #     tem_loss = test(Imagenet_val_loader, centers)
    #     # early stopping
    #     early_stopping(tem_loss, Encoder_on_device,Decoder_on_cloud,encoder_path,decoder_path, centers)
    #     if early_stopping.early_stop:
    #         print("Early stopping.")
    #         break
    # test(test_loader)
