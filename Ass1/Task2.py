import csv

MINSUP = 300

class DictValue(object):
    def __init__(self, frequency):
        self.frequency = frequency
        self.node_link = None


class TreeNode(object):
    def __init__(self, key, frequency, parent_node):
        self.key = key
        self.frequency = frequency
        self.node_link = None
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
                new_dic_value = DictValue(0)
                head_table[item_name] = new_dic_value
    for item_name in list(head_table):
        if head_table[item_name].frequency < MINSUP:
            head_table.pop(item_name)
    head_table = dict(sorted(head_table.items(), key=lambda x: x[1].frequency, reverse=True))
    return head_table


def compute_frequent_item(data, head_table):
    item_frequency_list = list(head_table)
    ret_data = []
    for row in data:
        new_row = []
        for item_name in row:
            if item_name in head_table:
                new_row.append(item_name)
        if len(new_row) != 0:
            row_sorted = [new_row for (new_row, item_frequency_list) in sorted(zip(new_row, item_frequency_list))]
            ret_data.append(row_sorted)
    return ret_data


def update_header(start_header_name, targetNode):
    while start_header_name.node_link is not None:
        start_header_name = start_header_name.node_link
    start_header_name.node_link = targetNode


def update_FPtree(frequent_item, root_node, head_table):
    if frequent_item[0] in root_node.children:
        root_node.children[frequent_item[0]].increase()
    else:
        root_node.children[frequent_item[0]] = TreeNode(frequent_item[0], 1, root_node)
        if head_table[frequent_item[0]].node_link is None:
            head_table[frequent_item[0]].node_link = root_node.children[frequent_item[0]]
        else:
            update_header(head_table[frequent_item[0]], root_node.children[frequent_item[0]])
    if len(frequent_item) > 1:
        update_FPtree(frequent_item[1::], root_node.children[frequent_item[0]], head_table)


def create_FPtree(frequent_item_table, head_table):
    root_node = TreeNode('Null', 1, None)
    for frequent_item in frequent_item_table:
        update_FPtree(frequent_item, root_node, head_table)
    return root_node


def ascend_FPtree(leaf_node, perfix_path):
    if leaf_node.parent is not None:
        perfix_path.append(leaf_node.key)
        ascend_FPtree(leaf_node.parent, perfix_path)


def find_prefix_path(head_table_item, head_table):
    cond_patten_list = []
    patten_item_frequency_set = {}
    sub_head_table = {}
    cur_node = head_table[head_table_item].node_link
    while cur_node is not None:
        prefix_path = []
        ascend_FPtree(cur_node, prefix_path)
        for item in prefix_path:
            if item in patten_item_frequency_set:
                patten_item_frequency_set[item] += 1
            else:
                patten_item_frequency_set[item] = 1
        if len(prefix_path) > 0:
            for num in range(cur_node.frequency):
                cond_patten_list.append(prefix_path)
        cur_node = cur_node.node_link
        for k,v in patten_item_frequency_set.items():
            if v > MINSUP:
                new_dict_value = DictValue(v)
                sub_head_table[k] = new_dict_value
        cond_patten_list = sorted(cond_patten_list)
        sub_head_table = dict(sorted(sub_head_table.items(), key=lambda x: x[1].frequency, reverse=True))
    return cond_patten_list, sub_head_table

def create_conditional_FPtree():
    pass


if __name__ == "__main__":

    data = csv_reader()

    candidate_patten_set = []

    head_table = compute_item_frequency(data)

    frequent_item_table = compute_frequent_item(data,head_table)

    root_node = create_FPtree(frequent_item_table, head_table)

    # root_node.display()

    for head_table_item in reversed(list(head_table)):
        cond_patten_list, sub_head_table = find_prefix_path(head_table_item, head_table)
        for item in sub_head_table:
            if len(item) == 2:
                candidate_patten_set.append(item)
            else:
                #big_recrusion(sub_head_table, cond_patten_list, candidate_patten_set)
                pass


'''
frequent_item_table: [
    ['soda', 'waffles']
    ['chocolate', 'root vegetables', 'sausage']
    ['UHT-milk', 'frozen vegetables', 'long life bakery product', 'root vegetables', 'whipped/sour cream', 'yogurt']
    ['hamburger meat', 'onions', 'rolls/buns', 'whole milk']
    ['bottled water', 'domestic eggs', 'fruit/vegetable juice', 'margarine', 'rolls/buns', 'tropical fruit', 'whipped/sour cream', 'whole milk']
]

cond_patten_list: [
    ['onions', 'cream cheese ', 'UHT-milk']
    ['onions', 'cream cheese ', 'bottled water']
    ['onions', 'curd']
    ['onions', 'curd']
    ['onions', 'curd']
    ['onions', 'curd', 'bottled water']
    ['onions', 'curd', 'citrus fruit', 'bottle']
]

head_table: {
    'whole milk': <__main__.DictValue object at 0x10c183198>, 
    'other vegetables': <__main__.DictValue object at 0x10c183278>, 
    'rolls/buns': <__main__.DictValue object at 0x10c1833c8>
}

sub_head_table: {
    'tropical fruit': <__main__.DictValue object at 0x10a01c048>,
    'other vegetables': <__main__.DictValue object at 0x109fe7f98>
}
'''