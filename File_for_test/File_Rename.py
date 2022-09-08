import os

'''
自動化改檔案名稱
'''


path = r'../Data/Pic/6'
number = 1
for old_name in os.listdir(path):
    old_path = os.path.join(path, old_name)
    new_name = '6_{}.jpg'.format(number)
    new_path = os.path.join(path, new_name)
    os.rename(old_path, new_path)
    print(old_name, '======>', new_name)
    number = number + 1


