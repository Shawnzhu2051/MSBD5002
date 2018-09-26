
data = [
    [1, 2, 4], [1, 2, 9], [1, 3, 5], [1, 3, 9], [1, 4, 7], [1, 5, 8], [1, 6, 7], [1, 7, 9],
    [1, 8, 9], [2, 3, 5], [2, 4, 7], [2, 5, 6], [2, 5, 7], [2, 5, 8], [2, 6, 7], [2, 6, 8],
    [2, 6, 9], [2, 7, 8], [3, 4, 5], [3, 4, 7], [3, 5, 7], [3, 5, 8], [3, 6, 8], [3, 7, 9],
    [3, 8, 9], [4, 5, 7], [4, 5, 8], [4, 6, 7], [4, 6, 9], [4, 7, 8], [5, 6, 7], [5, 7, 9],
    [5, 8, 9], [6, 7, 8], [6, 7, 9]
]

class HashTree(object):
    def __init__(self, index, depth):
        self.index = index
        self.leftChild = None
        self.midChild = None
        self.rightChild = None
        self.Container = []
        self.depth = depth

    def insertLeft(self):
        if self.leftChild is None:
            self.leftChild = HashTree(self.index+1, self.depth+1)

    def insertMid(self):
        if self.midChild is None:
            self.midChild = HashTree(self.index+1, self.depth+1)

    def insertRight(self):
        if self.rightChild is None:
            self.rightChild = HashTree(self.index+1, self.depth+1)

def TreeGenerate(hashtree):
    if hashtree.depth == 3:
        return
    if hashtree.leftChild == None:
        hashtree.insertLeft()
        TreeGenerate(hashtree.leftChild)
    if hashtree.midChild == None:
        hashtree.insertMid()
        TreeGenerate(hashtree.midChild)
    if hashtree.rightChild == None:
        hashtree.insertRight()
        TreeGenerate(hashtree.rightChild)

def scanTree(hashtree):
    if (len(hashtree.Container) != 0):
        print(hashtree.Container)
    if hashtree.leftChild == None:
        return
    else:
        scanTree(hashtree.leftChild)
    if hashtree.midChild == None:
        return
    else:
        scanTree(hashtree.midChild)
    if hashtree.rightChild == None:
        return
    else:
        scanTree(hashtree.rightChild)

if __name__ == "__main__":
    hashtree = HashTree(0,0)
    TreeGenerate(hashtree)
    for item in data:
        pointer = hashtree
        for pos in range(3):
            hash_result = item[pos-1] % 3
            if hash_result == 0:
                pointer = pointer.leftChild
            elif hash_result == 1:
                pointer = pointer.midChild
            elif hash_result == 2:
                pointer = pointer.rightChild
        pointer.Container.append(item)
    scanTree(hashtree)



