##################### huffman编码 ########################
import operator
import torch
# 统计字符出现频率，生成映射表
def count_frequency(text):
    chars = []
    ret = []

    for char in text:
        if char in chars:
            continue
        else:
            chars.append(char)
            ret.append((char, text.count(char)))
    ### [('a',4),.('b',4)..]
    # print('table:',ret)
    return ret

# 修改版，对tensor的最后一个维度进行编码
def count_frequency2(tensor_BCM):
    chars = {}
    ret = []

    for data in tensor_BCM:
        for x in data:
            if str(x.tolist()) in chars.keys():
                chars[str(x.tolist())] += 1
            else:
                chars[str(x.tolist())] = 1
    for x in chars.keys():
        ret.append((eval(x), chars[x]))
    return ret


# 节点类
class Node:
    def __init__(self, frequency):
        self.left = None
        self.right = None
        self.father = None
        self.frequency = frequency

    def is_left(self):
        return self.father.left == self


# 创建叶子节点
# def create_nodes(frequency_list):
#     return [Node(frequency) for frequency in frequency_list]


# 创建Huffman树
def create_huffman_tree(nodes):
    # 导入节点列表
    queue = nodes[:]

    while len(queue) > 1:
        # 按节点中对应的频率进行排序
        queue.sort(key=lambda item: item.frequency)
        # 删除并返回其中最小的的节点，用于构造叶子节点
        node_left = queue.pop(0)
        node_right = queue.pop(0)
        # 两个叶子节点的频率求和，构造其父节点
        node_father = Node(node_left.frequency + node_right.frequency)
        # 将之前得到的左右子节点接入父节点
        node_father.left = node_left
        node_father.right = node_right
        node_left.father = node_father
        node_right.father = node_father
        # 删除了两个节点，生成并导入了一个父节点-子节点的结构
        queue.append(node_father)
    # 根节点的父节点置为None
    queue[0].father = None
    # 返回根节点
    return queue[0]


# Huffman编码
def huffman_encoding(nodes, root):
    huffman_code = [''] * len(nodes)

    for i in range(len(nodes)):
        node = nodes[i]
        while node != root:
            if node.is_left():
                huffman_code[i] = '0' + huffman_code[i]
            else:
                huffman_code[i] = '1' + huffman_code[i]
            node = node.father

    return huffman_code


# 编码整个字符串
def encode_str(text, char_frequency, codes):
    ret = ''
    for char in text:
        for x in char:
            i = 0
            for item in char_frequency:
            # 字符串
            # if char == item[0]:
            # tensor
            # print(char)
            # print(item[0])
                if operator.eq(x.tolist(),item[0]):
                    ret += codes[i]
                # print('come in')
                i += 1

    return ret



# 编码整个字符串
def huffman_encode(text):
    # [B,C,w,h] → [B*C,w*h]
    text = text.view(text.shape[0]*text.shape[1],-1)
    ### 得到映射表
    char_frequency = count_frequency2(text)
    # print("char_frequency",char_frequency)
    ### 生成节点列表

    nodes = [Node(frequency) for frequency in [item[1] for item in char_frequency]]
    root = create_huffman_tree(nodes)
    codes = huffman_encoding(nodes, root)
    # print("codes",codes)
    huffman_str = encode_str(text, char_frequency, codes)
    return huffman_str, codes


# 解码整个字符串
def huffman_decode(huffman_str, centers, codes):
    ret = []
    while huffman_str != '':
        i = 0
        for item in codes:
            if item in huffman_str and huffman_str.index(item) == 0:
                ret.append(centers[i])
                huffman_str = huffman_str[len(item):]
            i += 1

    return torch.FloatTensor(ret)



# if __name__ == '__main__':
#     # text = 'The text to encode:'
#     import torch
#     # text = torch.FloatTensor([[1,2],[4,5],[1,2]])
#     text= torch.FloatTensor([[[[1],[2],[3]],[[4],[5],[6]],[[1],[2],[3]],[[7],[8],[9]]]])
#     centers = [1,2,3,4,5,6,7,8,9]
#     huffman_str, codes = huffman_encode(text)
#     origin_str = huffman_decode(huffman_str, centers, codes)
#     batch_num = text.shape[0]
#     print(text.shape)
#     origin_str = origin_str.view(batch_num,origin_str.shape[0]//(batch_num*text.shape[2]*text.shape[3]),text.shape[2],-1)
#     print(huffman_str)
#     print(origin_str)
#     print(text.equal(origin_str))
