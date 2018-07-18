import math
import numpy as np
np.set_printoptions(threshold=np.nan)

import binary_tree as bt

data = np.loadtxt('credit_approval.csv', dtype=str, delimiter=',')
data = np.delete(data,[1,2,7,10,13,14],axis=1)
# data = np.char.replace(data,'a','0')
# data = np.char.replace(data,'b','1')

## numpy array에 있는 문자들을 0부터 순서대로 int형으로 typecast를 해줌.
def convert_to_int(data):
    extract = []
    for i in range(len(data[0])):
        extract.append(set(data[:,i]))
    # print(extract)

    for i,hi in enumerate(extract):
        for num,word in enumerate(hi):
            # print(word,num)
            data[:,[i]] = np.char.replace(data[:,[i]],word,str(num))
    data = data.astype(int)
    return data

## colomn에 있는 문자의 개수를 추출함.
def extract_to_result_num_from_column(data,column_num=-1):
    return len(set(data[:,column_num]))

if __name__ == '__main__':
    print(extract_to_result_num_from_column(data,2))

def calculate_entropy(data):  ## data는 하나의 데이터와 결과가 필요함
    a = []
    attribute_num = extract_to_result_num_from_column(data,0)
    result_num = extract_to_result_num_from_column(data)
    for i in range(attribute_num):
        temp = []
        for j in range(result_num):
            temp.append(len(np.reshape(np.where((data[:,-1]==j)&(data[:,0]==i)),(-1))))
        a.append(temp)
    # print(a)
    result = 0
    sum = 0
    for j in a:
        for k in j:
            sum += k
    for j in a:
        sum_m = 0
        for k in j:
            sum_m += k
        for k in j:
            if k != 0:
                result += k/sum*math.log2(sum_m/k)
                # print("차례대로 결과값",result)
    return result

def calculate_information_gain(data):  ## data는 하나의 데이터와 결과가 필요함
    a = []
    attribute_num = extract_to_result_num_from_column(data,0)
    result_num = extract_to_result_num_from_column(data)
    for i in range(attribute_num):
        temp = []
        for j in range(result_num):
            temp.append(len(np.reshape(np.where((data[:,-1]==j)&(data[:,0]==i)),(-1))))
        a.append(temp)
    # print(a)
    result = 0
    sum = 0
    for j in a:
        for k in j:
            sum += k
    before = []
    for i in range(result_num):
        temp = 0
        for array in a:
            temp += array[i]
        before.append(temp)
    # print(before)
    H_y = 0
    for j in before:
        if j != 0:
            H_y += -j/sum * math.log2(j/sum)
    # print(H_y)
    for j in a:
        sum_m = 0
        for k in j:
            sum_m += k
        for k in j:
            if k != 0:
                result += k/sum*math.log2(sum_m/k)
                # print("차례대로 결과값",result)
    result = H_y - result
    # print(result,"hi")
    return result


data = convert_to_int(data)
print(data)
one_entropy = calculate_entropy(data[:,[0,-1]])
# print("one_entropy :", one_entropy)
entropy = []
for i in range(len(data[0])-1):
    sample = data[:,[i,-1]]
    entropy.append(calculate_information_gain(sample))
entropy = np.array(entropy)
print("entropy :",entropy)
tree = np.argmax(entropy,axis=0)
print("maximum :",tree ,":", entropy[tree] )

decision_tree = bt.Tree()
decision_tree.set_node(tree,extract_to_result_num_from_column(data,tree))
filter_data = []
for i in range(extract_to_result_num_from_column(data,tree)):
    filter_data.append(np.reshape(np.where((data[:,5]==i)),(-1)))
    # filter_data.append(np.extract(data[:,5]==i,data))
    processed_data = data[filter_data[i]]

    # print("filter_data :", filter_data[i],(-1,len(data[0])))
    entropy = []
    for j in range(len(data[0])-1):
        sample = processed_data[:,[j,-1]]
        entropy.append(calculate_information_gain(sample))
    entropy = np.array(entropy)
    print("entropy :",entropy)
    tree = np.argmax(entropy,axis=0)
    print("maximum :",tree ,":", entropy[tree] )
    decision_tree.put_leaf_deeply([i],i,tree,extract_to_result_num_from_column(processed_data,tree))
    print(decision_tree.root.leaves[i].value)
    print("----------------------------------------------")
    filter2_data = []
    print("filter2" , filter2_data)
    for j in range(extract_to_result_num_from_column(processed_data,tree)):
        filter2_data.append(np.reshape(np.where((processed_data[:,decision_tree.root.leaves[i].value]==j)),(-1)))
        # filter2_data.append(np.extract(data[:,5]==i,data))
        second_data = processed_data[filter2_data[j]]
        entropy = []
        for k in range(len(data[0])-1):
            sample = second_data[:,[k,-1]]
            entropy.append(calculate_information_gain(sample))
        entropy = np.array(entropy)
        print("entropy :",entropy)
        tree = np.argmax(entropy,axis=0)
        print("maximum :",tree ,":", entropy[tree] )
        decision_tree.put_leaf_deeply([i,j],j,tree,extract_to_result_num_from_column(second_data,tree))
        print(decision_tree.root.leaves[i].leaves[j].value)
    print("-------------finish--------------------")

        # print("filter_data :", filter_data[i],(-1,len(data[0])))


plus = [307,383]
a = [[98,112],[206,262],[3,9]]
t = [[284,77],[23, 306]]
sum=0
for i in plus:
    sum += i
H_y = 0
for i in plus:
    H_y += -i/sum*math.log2(i/sum)
print("before entopy :",H_y)

result=0
# for i in plus:
for j in a:
    sum_m = 0
    for k in j:
        sum_m += k
    # print(sum_m)
    for k in j:
        result += k/sum*math.log2(sum_m/k)
result = H_y -result
print(result)
result=0

for j in t:
    sum_m = 0
    for k in j:
        sum_m += k
    for k in j:
        result += k/sum*math.log2(sum_m/k)
print(result)
result = H_y -result

print("information Gain :",result)
