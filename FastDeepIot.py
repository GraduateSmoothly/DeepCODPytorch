from scipy import stats
from sklearn.metrics import mean_squared_error # 均方误差
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

# 返回满足等式三的最优条件
def equation_two(D,condition_list):
    temp = []
    for condition in condition_list:
        temp.append(equation_three(D,condition))
    infer_index = temp.index(min(temp))
    return condition_list[infer_index]

# impurity function
def equation_three(D,condition):
    condition = True
    dl = []
    dr = []
    if condition:
        dl.append()
    else:
        dr.append()
    return (len(dl)*equation_four(dl)+len(dr)*equation_four(dr) ) /len(D)


# 对D算均方误差
def equation_four(D):
    x,y = read_from_D(D)
    res = stats.linregress(x,y)
    y_predict = res.intercept + res.slope * x
    return mean_squared_error(y_predict,y)

# 从D中读出x和y
def read_from_D(D):
    pass