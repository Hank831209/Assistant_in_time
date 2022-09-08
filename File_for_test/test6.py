import shutil
import os

path = r'C:/Users/TibeMe_user/Desktop/Resize_0905/0'
try:
    shutil.rmtree(path)
except OSError as e:
    print(e)
else:
    print("The directory is deleted successfully")

if os.path.isdir(path):
  print("目錄存在。")
else:
  print("目錄不存在。")
