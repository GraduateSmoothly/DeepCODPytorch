import datetime
from socket import *
import torch
from torchvision.models import resnet50
import socket

from Huffman_encode_and_decode import *
### 规定运行设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
### 网络部分
def net_part(image, model_name):
    if model_name == 'resnet50':
        model = resnet50(pretrained=True, progress=True)
        model = model.to(device)
    return model(image)

def connect_to_client(HOST, PORT, centers, decode_model):
    decode_model = decode_model.to(device)
    # 2 定义缓冲区(缓存)
    BUFFER_SIZE = 1024

    # hostname = socket.gethostname()
    # HOST = socket.gethostbyname(hostname)

    print('HOST',HOST)
    ADDR = (HOST, PORT)
    # 3 创建服务器套接字 AF_INET:IPv4  SOCK_STREAM:协议
    tcpServerSocket = socket.socket(AF_INET, SOCK_STREAM)
    # 4 绑定域名和端口号
    tcpServerSocket.bind(ADDR)
    # 5 监听连接最大连接数
    tcpServerSocket.listen(5)
    # 6 定义一个循环 目的:等待客户端的连接
    while True:
        # 6.1 打开一个客户端对象 同意连接
        print('*' * 20, 'begin', '*' * 20)
        tcpClientSocket, addr = tcpServerSocket.accept()
        # print(addr)
        temp = ''
        while True:
            test1 = ""
            # 6.2 接受数据
            print('*' * 20, 'connect', '*' * 20)
            data = tcpClientSocket.recv(BUFFER_SIZE)
            # 6.3 数据复原
            data = data.decode('utf-8')
            data = data
            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            start1 = datetime.datetime.now()
            if data != None:
                temp += data
                # print(len(temp))
                if data[-1] == ']' and data[-2] == ']':
                    huffman_out, codes, outshape = eval(temp)
                    temp = ''
                else:
                    continue
                # # 6.3 解码huffman数据
            data_decode = huffman_decode(huffman_out[1:], centers, codes)
            data_decode = data_decode.to(device)
            end1 = datetime.datetime.now()
            print('Huffman decode results:',end1-start1)
            # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            print('data_decode', data_decode.shape)
                # [1,N*C*w*h] → [N,C,w,h]
            origin_str = data_decode.view(outshape[0],
                                              outshape[1],
                                              outshape[2],
                                              outshape[3])
                # # 6.4 经过decoder进行数据解码
            out = decode_model(origin_str)
            end2 = datetime.datetime.now()
            print('Decoder results:',end2-end1)
            # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            # # 6.5 解码后数据传入网络后半部分
            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            results = net_part(out,'resnet50')
            predict = list(results.argmax(dim=1, keepdim=True))
            end3 = datetime.datetime.now()
            print('Net results:',end3-end2)
            # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            # results = 'I am the sever'
            # 返回结果给客户端
            tcpClientSocket.send(str(predict).encode('utf-8'))
            # tcpClientSocket.send(str(results).encode('utf-8'))
            # print(results)

        # 7 关闭资源
            tcpClientSocket.close()
            break

    tcpServerSocket.close()
