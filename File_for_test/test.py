import cv2


def close_center(h, s, v):
    black = (90, 122.5, 23)
    gray = (90, 21.5, 133)
    white = (90, 15, 238)
    red1 = (10, 149, 150.5)
    red2 = (168, 149, 150.5)
    orange = (18, 149, 150.5)
    yellow = (30, 149, 150.5)
    green = (56, 149, 150.5)
    blue_ = (88.5, 149, 150.5)
    blue = (112, 149, 150.5)
    purple = (140, 149, 150.5)



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

    # if color == 'None'

    return color


# for h in range(181):
#     for s in range(256):
#         for v in range(256):
#             ans = color_check(h, s, v)
#             if ans == 'None':
#                 print(h, s, v)








