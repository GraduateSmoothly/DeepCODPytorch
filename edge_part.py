import os
from Decoder_Edge_Server import *
from sever_socket import  *
from configuration import *
config = parser.parse_args()
# 给定解码器的路径
decoder_model_path = ''
# 给出解码器输入维度.这里等于编码前的输出维度，本应该是fastdeepiot来确定的，默认为32
in_dim = max(int(3 * config.compress_ratio * 4 * 4), 1)
# 设立编码器输出维度，本应该是fastdeepiot来确定的，这里默认为32
out_dim = 3
# 加载预训练好的解码器
decoder_path = r'D:\DAstudy\import_baseline_few-shot\DeepCODPytorch\endecoder_save\best_decoder.pt'
Decoder_on_cloud = Decoder_edge(in_dim, out_dim)
Decoder_on_cloud.load_state_dict(torch.load(decoder_path))



# 导入预训练好的centers
centers_path = r'D:\DAstudy\import_baseline_few-shot\DeepCODPytorch\endecoder_save\centers.txt'
# regu_param_path = r'D:\DAstudy\import_baseline_few-shot\DeepCODPytorch\endecoder_save\regu_param.txt'
if os.path.exists(centers_path):
    with open(centers_path, 'r') as f1:
        centers = torch.Tensor(eval(f1.read()))
    f1.close()
# 连接客户端，其他操作在连接后处理
# 给定一个空闲端口，需要和客户端给定的一样
HOST = '192.168.3.248'
PORT = 8888
connect_to_client(HOST, PORT, centers, Decoder_on_cloud)

