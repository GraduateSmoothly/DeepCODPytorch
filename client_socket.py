import socket
import torch
import datetime
def connect_to_server(HOST,PORT, data):
    # 2.创建套接字对象，AF_INET基于TCP/UDP通信，SOCK_STREAM以数据流的形式传输数据，这里就可以确定是TCP了
    client = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
    BUFFER_SIZE = 1024
    ### 得到计算机的ip及空闲端口
    print('*' * 20, 'begin connect', '*' * 20)
    # 3.连接服务端
    client.connect((HOST, PORT))
    print('*' * 20, 'connect success', '*' * 20)
    while True:
        with torch.autograd.profiler.profile(use_cpu=True) as prof:
            begin = datetime.datetime.now()
            # 4.确定要发送的数据, (huffman编码序列, centers, codes, 编码前的shape)
            temp_data = str(data)
            print(len(temp_data))
            # 如果没有要传输的数据，则关闭循环
            if not temp_data:
                break
            print('*' * 20, 'send data', '*' * 20)
            # 5.向服务端发送数据，需要转换成Bytes类型发送
            client.send(temp_data.encode('utf-8'))

            print('send end')
            back_data = client.recv(BUFFER_SIZE)
            print(back_data.decode('utf-8'))
            end = datetime.datetime.now()
        print(end-begin)
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        print('Get results:',prof)
            # # 数据清空
            # temp_data = None
        print('*' * 20, 'finished', '*' * 20)
        # 套接字关闭
        client.close()
        break

