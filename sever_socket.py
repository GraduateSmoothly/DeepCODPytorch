from socket import *
import csv
import os
import numpy as np
import matplotlib.pyplot  as plt
from scipy.interpolate import interpolate
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers

storeDir = "D:\\pycharmProject\\PAWrite\\PAWrite\\data\\experiment\\10number\\7\\"
keyWord = ""
number = 1

homePath = "C:\\Users\\15022\\PycharmProjects\\Server\\"

labelList = ["+", "D", "G", "M", "U"]
numberList = ["1", "2", "3", "4", "5"]
letterList = ["a", "b", "c", "d", "e"]

resultList = letterList


def normalization(path):
    ma = max(path)
    mi = min(path)
    interval = ma - mi
    result = []
    for i in path:
        result.append((i - mi) / interval)
    return result


# 写CSV
def writeFile(file, data):
    f = open(file, 'w', encoding='utf-8', newline='')
    csv_w = csv.writer(f)
    for i in data:
        csv_w.writerow([str(i)])
    f.close()


def quTou2(data, threshold=10):
    result = data

    #     去头去尾
    i = 1
    tou = result[0]
    wei = result[-1]
    while i < len(result) - 1:
        if abs(result[i] - tou) < threshold:
            del result[i]
        else:
            break
    # 去尾
    weiThreshold = threshold
    i = len(result) - 2
    while i > 0:
        if abs(result[i] - wei) < weiThreshold:
            del result[i]
        else:
            break
        i -= 1

    return result


def changeLength(list, lenght):
    x = range(1, len(list) + 1, 1)
    x = np.array(x)
    y = np.array(list)
    f_linear = interpolate.interp1d(x, y, kind="linear")
    new_x = x = np.linspace(x[0], x[-1], lenght)
    new_y = f_linear(new_x)
    return new_y


model = keras.models.load_model("best_model_letter.h5")

test1 = ""
# 1 定义域名和端口号
HOST, PORT = '192.168.2.26', 12345
# 2 定义缓冲区(缓存)
BUFFER_SIZE = 1024
ADDR = (HOST, PORT)
# 3 创建服务器套接字 AF_INET:IPv4  SOCK_STREAM:协议
tcpServerSocket = socket(AF_INET, SOCK_STREAM)
# 4 绑定域名和端口号
tcpServerSocket.bind(ADDR)
# 5 监听连接  最大连接数
tcpServerSocket.listen(5)
# 6 定义一个循环 目的:等待客户端的连接
while True:
    # 6.1 打开一个客户端对象 同意连接
    tcpClientSocket, addr = tcpServerSocket.accept()
    print(addr)
    while True:
        test1 = ""
        # 6.2 接受数据
        data = tcpClientSocket.recv(BUFFER_SIZE)
        # 6.3 解码数据
        print(data)

        # data2 = data.decode('utf-8')
        # data3 = data2.split(" ")
        # data4 = []
        # for i in data3:
        #     data4.append(int(i))
        # # data4里面是数据
        # writeFile( storeDir + keyWord + str(number) + ".csv", data4 )
        # number += 1
        #
        #
        #
        #
        # plt.cla()
        # # plt.figure()
        # plt.plot( changeLength( normalization(data4),256), '-b', linewidth=5.0)
        #
        # data4 = quTou2((data4))
        # data4 = normalization(data4)
        # data = changeLength(data4 , 256)
        #
        #
        # plt.plot(data , '-r' ,  linewidth=5.0)
        # plt.show()
        #
        # data = np.array([data])
        # predictResult =np.argmax( model.predict(data) , axis=1)
        #
        #
        # print('result', resultList[predictResult[0] ])
        break
    tcpClientSocket.send(test1.encode())
    # 7 关闭资源
    tcpClientSocket.close()
tcpServerSocket.close()
