import socket
# 1 定义域名和端口号
HOST, PORT = '192.168.2.26', 12345
# 2.创建套接字对象，AF_INET基于TCP/UDP通信，SOCK_STREAM以数据流的形式传输数据，这里就可以确定是TCP了
client = socket.socket(family=socket.AF_INET,type=socket.SOCK_STREAM)

# 3.连接服务端
client.connect((HOST,PORT))
while True:
    # 4.确定要发送的数据
    data = 'raw_input()'
    # 如果没有要传输的数据，则关闭循环
    if not data:
        break
    # 5.向服务端发送数据，需要转换成Bytes类型发送
    client.send('Hello'.encode('utf-8'))
    # 数据清空
    data = None

# 套接字关闭
client.close()
