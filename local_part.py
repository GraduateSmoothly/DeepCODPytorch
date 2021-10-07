import time

from PIL import Image, ImageDraw

from Huffman_encode_and_decode import huffman_encode
from encoder_mobile import *
from client_socket import  *
from configuration import *
from quantization_part import *
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from configuration import *
import torch

### 编码器部分
sigma = 1  ## for smooth
# 设置数据格式
data_format = 'NCHW'
# 导入参数
config = parser.parse_args()
# 初始化核的大小
kernel_size = 4
# 给出编码器输入维度,即前半段网络输出的维度
in_channels = 3
# 导入预训练好的centers
centers_path = r'D:\DAstudy\import_baseline_few-shot\DeepCODPytorch\endecoder_save\centers.txt'
if os.path.exists(centers_path):
    with open(centers_path, 'r') as f1:
        centers = torch.Tensor(eval(f1.read()))
    f1.close()
# 设立编码器输出维度，本应该是fastdeepiot来确定的
out_channels = max(int(3 * config.compress_ratio * 4 * 4), 1)
encoder_path = r'D:\DAstudy\import_baseline_few-shot\DeepCODPytorch\endecoder_save\best_encoder.pt'
Encoder_on_device = encoder_mobile_part(in_channels, out_channels, kernel_size)
Encoder_on_device.load_state_dict(torch.load(encoder_path))

def test_data_in(app_mode,test_image_name):
    if app_mode == 'image':
        # 数据存储路径
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # valdir = os.path.join(r'D:\DAstudy\import_baseline_few-shot\DeepCODPytorch\dataset\ImageNet_minitest', 'test')
        # test_data_loader = torch.utils.data.DataLoader(
        #     datasets.ImageFolder(valdir, transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         normalize,
        #     ])),
        #     batch_size=config.batch_size, shuffle=False,
        #     num_workers=config.workers, pin_memory=True)
        # print('test_data_loader: ',len(test_data_loader))
        # average_time = 0

        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        with torch.autograd.profiler.profile() as prof:
            test_image = Image.open(test_image_name)
            # draw = ImageDraw.Draw(test_image)
            data = transform(test_image).view(1, 3, 224, 224)
            # data = test_image_tensor.view(1, 3, 224, 224)
        print('Preproccess time:', prof.key_averages().table(sort_by="self_cpu_time_total"))

        with torch.no_grad():
            Encoder_on_device.eval()



        # 调用编码器得到编码结果
        # for data, label in test_data_loader:
            # print(len(data))

            with torch.autograd.profiler.profile() as prof:
                ## 将数据导入网络模型
                encode_out = Encoder_on_device(data)
            # average_time += prof.self_cpu_time_total / config.batch_size
            print('Encoder time:', prof.key_averages().table(sort_by="self_cpu_time_total"))

            with torch.autograd.profiler.profile() as prof:
                out = center_quatization(encode_out, centers, sigma, data_format, 'cpu')
            print('Center time:', prof.key_averages().table(sort_by="self_cpu_time_total"))
            with torch.autograd.profiler.profile() as prof:

                ### 下面接huffman编码
                out_text = out.view(out.shape[0] * out.shape[1], -1)
                huffman_out, codes = huffman_encode(out_text)
                # print(len(huffman_out))
                outshape = list(out.shape)
                # print(outshape)
                ### 确定传输数据
                transfer_data = [huffman_out, codes, outshape]

                # print(transfer_data)
                ### 连接服务器，传输经过huffman编码的数据
                # 1 定义域名和端口号,电脑的ip
                # PORT = 8888

            print('Huffman time:', prof.key_averages().table(sort_by="self_cpu_time_total"))
            return
            connect_to_server(HOST, PORT, transfer_data)
            # break
        # print('The average time for propossing one image is:{:.4f}'.format(average_time/len(test_data_loader)))

if __name__ == '__main__':

    HOST = '192.168.3.248'
    PORT = 8888
    test_image_name = r'D:\DAstudy\import_baseline_few-shot\DeepCODPytorch\dataset\ImageNet_minitest\val\box\n0297135600000208.jpg'
    test_data_in('image',test_image_name)
    # temp = 'I am the client'
    # connect_to_server(HOST,PORT, temp)
