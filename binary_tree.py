import numpy as np

class Node:
    def __init__(self,value,leaves_num=0):
        self.value = value
        self.leaves_num = leaves_num
        self.leaves = [None for i in range(leaves_num)]

    def set_value(self,value):
        self.value = value

    def set_leaves_num(self,leaves_num):
        self.leaves_num = leaves_num
        self.leaves = [None for i in range(leaves_num)]

    def set_leaf(self,num,value,leaves_num=0):
        self.leaves[num] = Node(value,leaves_num)



class Tree:
    def __init__(self):
        self.root = None

    def set_node(self,value,leaves_num=0):
        self.root = Node(value,leaves_num)
        print("leaves",self.root.leaves_num)

    def put_leaf(self,num,value,leaves_num=0):
        self.root.set_leaf(num,value,leaves_num)

    def put_leaves(self,final_num,values,leaves_nums=0):
        self.root.set_leaves_num(final_num)
        for i in range(final_num):
            self.root.set_leaf(i,values[i],leaves_nums[i])

    def put_leaf_deeply(self,location,num,value,leaves_num=0):
        iterator = None
        for i,locate in enumerate(location):
            if i==0:
                if self.root.leaves[locate]==None:
                    iterator = self.root.leaves[locate]
                    iterator = Node(value,leaves_num)
                    self.root.set_leaf(num,value,leaves_num)
                iterator = self.root.leaves[locate]
            else:
                if iterator.leaves[locate]==None:
                    iterator.leaves[locate] =Node(value,leaves_num)
                    iterator.set_leaf(num,value,leaves_num)
                iterator = iterator.leaves[locate]

    def put_leaves_deeply(self,location,final_num,values,leaves_nums):
        iterator = None
        for i,locate in enumerate(location):
            if i==0:
                iterator = self.root.leaves[locate]
            else:
                iterator = iterator.leaves[locate]
        iterator.set_leaves_num(final_num)
        for i in range(final_num):
            iterator.set_leaf(i,values[i],leaves_nums[i])

if __name__ == "__main__":
    hi = Tree()
