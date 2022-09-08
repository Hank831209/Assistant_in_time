import os
import shutil
import random
# 0: 3585
# 1: 2671
# 2: 771
# 3: 1992
# 4: 1134
# 5: 787
tran_class = '0'  # 第幾個class資料
root = r'../Data/Pic/Resize_0907B'  # 資料路徑
path_Raw_Data = os.path.join(root, 'Raw_Data')  # 原始資料路徑
train, test = os.path.join('train', tran_class), os.path.join('test', tran_class)
path = os.path.join(path_Raw_Data, tran_class)
spilt_set = [train, train, train, train, train, train, train, test, test, test]  # 73分
transform_path = os.path.join(root, 'Split_Data')  # 分類的資料路徑
for old_name in os.listdir(path):
    old_path = os.path.join(path, old_name)
    decide_set, new_name = spilt_set[random.randrange(10)], old_name
    new_path = os.path.join(transform_path, decide_set, new_name)
    shutil.copy(old_path, new_path)
    print(old_name, '======>', new_path)

path_train = os.path.join(transform_path, train)
path_test = os.path.join(transform_path, test)
a = len(os.listdir(path_train))
b = len(os.listdir(path_test))
c = a + b
print('總資料量:\t', c)
print('訓練集資料量:\n', a, '\t資料占比:\t', a/c)
print('測試集資料量:\n', b, '\t資料占比:\t', b/c)



