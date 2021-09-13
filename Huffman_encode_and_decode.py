##################### huffman编码 ########################
import operator
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
        if str(data.tolist()) in chars.keys():
            chars[str(data.tolist())] += 1
        else:
            chars[str(data.tolist())] = 1
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
def create_nodes(frequency_list):
    return [Node(frequency) for frequency in frequency_list]


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
        i = 0
        for item in char_frequency:
            # 字符串
            # if char == item[0]:
            # tensor
            # print(char)
            # print(item[0])
            if operator.eq(char.tolist(),item[0]):
                ret += codes[i]
                # print('come in')
            i += 1

    return ret


# 解码整个字符串
def decode_str(huffman_str, char_frequency, codes):
    ret = []
    while huffman_str != '':
        i = 0
        for item in codes:
            if item in huffman_str and huffman_str.index(item) == 0:
                ret.append(char_frequency[i][0])
                huffman_str = huffman_str[len(item):]
            i += 1

    return torch.FloatTensor(ret)

def huffman_encode_decode(text):
    text = text.view(text.shape[0]*text.shape[1],-1)
    ### 得到映射表
    char_frequency = count_frequency2(text)
    # print(char_frequency)
    ### 生成节点列表
    nodes = create_nodes([item[1] for item in char_frequency])
    root = create_huffman_tree(nodes)
    codes = huffman_encoding(nodes, root)
    # print(codes)
    huffman_str = encode_str(text, char_frequency, codes)
    origin_str = decode_str(huffman_str, char_frequency, codes)
    return huffman_str, origin_str


if __name__ == '__main__':
    # text = 'The text to encode:'
    import torch
    # text = torch.FloatTensor([[1,2],[4,5],[1,2]])
    text= torch.rand(2,2,3)
    huffman_str, origin_str = huffman_encode_decode(text)
    origin_str = origin_str.view(text.shape[0],text.shape[1],-1)
    print(origin_str)
    print(text.equal(origin_str))
