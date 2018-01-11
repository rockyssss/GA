# _*_ coding: UTF-8 _*_

import numpy as np

#求下一个房间坐标的代码
# list = [[34, 56], [43, 78], [26, 43]]

def cross(x, y, ListLW):
    ListResult = []
    for i in ListLW:
        x= x + i[0]

        y = i[1]
        ListResult.append([x, y])
    return ListResult
# ll = cross(0, 0, list)
# print(cross(0, 0, list))

#vertical垂直
def vertical(x, y, ListLW):
    ListResult = []
    for i in ListLW:
        x = i[0]
        y = y + i[1]
        ListResult.append([x, y])
    return ListResult
# print(ll[2][0])
# print(ll)
# print(vertical(ll[2][0], ll[2][1], ll))
def crossSingle(x, y, ListLW, direction):
    if (direction.dx != 0):
        x = x + ListLW[0] * direction.dx
    if (direction.dy != 0):
        y = y + ListLW[0] * direction.dy


    # y = ListLW[1]
    ListResult=[x, y]
    return ListResult
def verticalSingle(x, y, ListLW):

    y = y + ListLW[1]
    ListResult=[x, y]
    return ListResult