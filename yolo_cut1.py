import cv2
import os

capture = cv2.VideoCapture(0)
# 判断摄像头是否可用
# 若可用，则获取视频返回值ref和每一帧返回值frame
if capture.isOpened():
    ref, frame = capture.read()
else:
    ref = False

# 间隔帧数
timeF = 200

c = 1
while ref:
    ref, frame = capture.read()

    # 每隔timeF获取一张图片并保存到指定目录
    # "D:/photo/"根据自己的目录修改
    if c % timeF == 0:
        cv2.imwrite(r"C\Desktop\SHIXUN\yolov5-v5.0\runs\detect\exp\labels\chepai.jpg", frame)
    # break
    c += 1
    # 1毫秒刷新一次
    k = cv2.waitKey(1)

# 按q退出
# if k==27：则为按ESC退出

path = r"C:\UDesktop\SHIXUN\yolov5-v5.0\runs\detect\exp\labels"  # jpg图片和对应的生成结果的txt标注文件，放在一起
path3 = r"C:\UsDesktop\SHIXUN\yolov5-v5.0\runs\detect\exp"  # 裁剪出来的小图保存的根目录
w = 1000  # 原始图片resize
h = 1000
img_total = []
txt_total = []

file = os.listdir(path)
for filename in file:
    first, last = os.path.splitext(filename)
    if last == ".jpg":  # 图片的后缀名
        img_total.append(first)
        # print(img_total)
    else:
        txt_total.append(first)

for img_ in img_total:
    if img_ in txt_total:
        filename_img = img_ + ".jpg"  # 图片的后缀名
        # print('filename_img:', filename_img)
    path1 = os.path.join(path, filename_img)
    img = cv2.imread(path1)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)  # resize 图像大小，否则roi区域可能会报错
    filename_txt = img_ + ".txt"
    # print('filename_txt:', filename_txt)
n = 1
with open(os.path.join(path, filename_txt), "r+", encoding="utf-8", errors="ignore") as f:
    for line in f:
        aa = line.split(" ")
        x_center = w * float(aa[1])  # aa[1]左上点的x坐标
y_center = h * float(aa[2])  # aa[2]左上点的y坐标
width = int(w * float(aa[3]))  # aa[3]图片width
height = int(h * float(aa[4]))  # aa[4]图片height
lefttopx = int(x_center - width / 2.0)
lefttopy = int(y_center - height / 2.0)
roi = img[lefttopy + 1:lefttopy + height + 3, lefttopx + 1:lefttopx + width + 1]  # [左上y:右下y,左上x:右下x] (y1:y2,x1:x2)需要调参，否则裁剪出来的小图可能不太好
print('roi:', roi)  # 如果不resize图片统一大小，可能会得到有的roi为[]导致报错
filename_last = img_ + "_" + str(n) + ".jpg"  # 裁剪出来的小图文件名
print(filename_last)
                path2 = os.path.join(path3, "roi")  # 需要在path3路径下创建一个roi文件夹
print('path2:', path2)  # 裁剪小图的保存位置
cv2.imwrite(os.path.join(path3, filename_last), roi)
                n = n + 1
        os.remove(
 r'C:\Users\esktop\SHIXUN\yolov5-v5.0\runs\detect\exp\labels\chepai.txt')
 else:
 continue