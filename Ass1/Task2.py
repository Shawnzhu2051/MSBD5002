import csv

MINSUP = 300

class DictValue(object):
    def __init__(self):
        self.frequency = 0
        self.link2tree = None


class TreeNode(object):
    def __init__(self, key, frequency, parent_node):
        self.key = key
        self.frequency = frequency
        self.nodeLink = None
        self.parent = parent_node
        self.children = {}

    def increase(self):
        self.frequency += 1

    def display(self, ind=1):
        print('  '*ind, self.key, ' ', self.frequency)
        for child in self.children.values():
            child.display(ind+1)


def csv_reader():
    data = []
    with open('groceries.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            line = []
            for item in row:
                if item != '':
                    line.append(item)
            data.append(line)
    return data


def compute_item_frequency(data):
    head_table = {}
    for row in data:
        for item_name in row:
            if item_name in head_table:
                head_table[item_name].frequency += 1
            else:
                new_dic_value = DictValue()
                head_table[item_name] = new_dic_value
    for item_name in list(head_table):
        if head_table[item_name].frequency < 300:
            head_table.pop(item_name)
    return head_table


def compute_frequent_item(data, head_table):
    item_frequency_list = list(head_table)
    ret_data = []
    for row in data:
        for item_name in row:
            if item_name not in head_table:
                row.remove(item_name)
        row_sorted = [row for (row, item_frequency_list) in sorted(zip(row, item_frequency_list))]
        if len(row_sorted) != 0:
            ret_data.append(row_sorted)
    return ret_data

'''
def updateHeader(nodeToTest, targetNode):
    while nodeToTest.nodeLink != None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode
    
def updateFPtree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        # 判断items的第一个结点是否已作为子结点
        inTree.children[items[0]].inc(count)
    else:
        # 创建新的分支
        inTree.children[items[0]] = TreeNode(items[0], count, inTree)
        # 更新相应频繁项集的链表，往后添加
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    # 递归
    if len(items) > 1:
        updateFPtree(items[1::], inTree.children[items[0]], headerTable, count)
'''
def update_Header(nodeToTest, targetNode):
    pass

def update_FPtree(frequent_item, root_node, head_table):
    if frequent_item[0] in root_node.children:
        root_node.children[frequent_item[0]].increase()
    else:
        pass
    if len(frequent_item) > 1:
        update_FPtree(frequent_item[1::], root_node.children[frequent_item[0]], head_table)

'''
def createFPtree(dataSet, minSup=1):
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in headerTable.keys():
        if headerTable[k] < minSup:
            del(headerTable[k]) # 删除不满足最小支持度的元素
    freqItemSet = set(headerTable.keys()) # 满足最小支持度的频繁项集
    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None] # element: [count, node]

    retTree = TreeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        # dataSet：[element, count]
        localD = {}
        for item in tranSet:
            if item in freqItemSet: # 过滤，只取该样本中满足最小支持度的频繁项
                localD[item] = headerTable[item][0] # element : count
        if len(localD) > 0:
            # 根据全局频数从大到小对单样本排序
            orderedItem = [v[0] for v in sorted(localD.items(), key=lambda p:p[1], reverse=True)]
            # 用过滤且排序后的样本更新树
            updateFPtree(orderedItem, retTree, headerTable, count)
    return retTree, headerTable
'''

def create_FPtree(frequent_item_table,head_table,MINSUP):
    root_node = TreeNode('Null', 1, None)
    for frequent_item in frequent_item_table:
        update_FPtree(frequent_item, root_node, head_table)

if __name__ == "__main__":
    data = csv_reader()
    head_table = compute_item_frequency(data)
    head_table = dict(sorted(head_table.items(), key=lambda x: x[1].frequency, reverse=True))
    frequent_item_table = compute_frequent_item(data,head_table)



