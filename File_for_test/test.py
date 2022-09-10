import cv2


def close_center(h, s, v):
    # print('執行算距離程式')
    color = {
        '黑': (90, 122.5, 23),
        '灰': (90, 21.5, 133),
        '白': (90, 15, 238),
        '紅1': (10, 149, 150.5),
        '紅2': (168, 149, 150.5),
        '澄': (18, 149, 150.5),
        '黃': (30, 149, 150.5),
        '綠': (56, 149, 150.5),
        '青': (88.5, 149, 150.5),
        '藍': (112, 149, 150.5),
        '紫': (140, 149, 150.5)
    }
    global_min = 257*257*257
    min_color = 'None'
    for key in color.keys():
        distance = ((h - color[key][0])**2 + (s - color[key][1])**2 + (v - color[key][2])**2)**0.5
        if distance <= global_min:
            min_color = key
            global_min = distance  # 最小距離

    if min_color == '紅1' or '紅2':
        return '紅'
    else:
        return min_color


def color_check(H, S, V):
    color = 'None'
    if (0 <= H <= 180) & (0 <= S <= 255) & (0 <= V < 46):  # 90, 122.5, 23
        color = '黑'
    elif (0 <= H <= 180) & (0 <= S <= 43) & (46 <= V <= 220):  # 90, 21.5, 133
        color = '灰'
    elif (0 <= H <= 180) & (0 <= S <= 30) & (221 <= V <= 255):  # 90, 15, 238
        color = '白'
    elif (0 <= H <= 10 or 156 <= H <= 180) & (43 <= S <= 255) & (46 <= V <= 255):  # (10, 168), 149, 150.5
        color = '紅'
    elif (11 <= H <= 25) & (43 <= S <= 255) & (46 <= V <= 255):  # 18, 149, 150.5
        color = '澄'
    elif (26 <= H <= 34) & (43 <= S <= 255) & (46 <= V <= 255):  # 30, 149, 150.5
        color = '黃'
    elif (35 <= H <= 77) & (43 <= S <= 255) & (46 <= V <= 255):  # 56, 149, 150.5
        color = '綠'
    elif (78 <= H <= 99) & (43 <= S <= 255) & (46 <= V <= 255):  # 88.5, 149, 150.5
        color = '青'
    elif (100 <= H <= 124) & (43 <= S <= 255) & (46 <= V <= 255):  # 112, 149, 150.5
        color = '藍'
    elif (125 <= H <= 155) & (43 <= S <= 255) & (46 <= V <= 255):  # 140, 149, 150.5
        color = '紫'

    if color == 'None':
        color = close_center(H, S, V)

    return color


for h in range(181):
    for s in range(256):
        for v in range(256):
            ans = color_check(h, s, v)
            if ans == 'None':
                print(h, s, v)
print('程式結束')








