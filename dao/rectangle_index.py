# _*_ coding: UTF-8 _*_
"""
示例:
若a = np.array([0, 1, 2, 3, 4, 5, 2, -0.25, -0.5, -0.25])
此时则有：
a[X_INDEX]为0---------------矩形a 左下角点x坐标为0
a[Y_INDEX]为1---------------矩形a 左下角点y坐标为1
a[L_INDEX]为2---------------矩形a 长度为2
a[W_INDEX]为3---------------矩形a 宽度为3
a[SPACE_ID_INDEX]为4--------矩形a 对应的space_id为4
a[TYPE_INDEX]为5------------矩形a 对应的type为5
a[B1_INDEX]为2--------------矩形a B1边邻接的房间为2号房间
a[B2_INDEX]为-0.25-----------矩形a B2边邻接走廊的概率为0.25
a[B3_INDEX]为-0.5------------矩形a B3边邻接走廊的概率为0.5
a[B4_INDEX]为-0.25-----------矩形a B4边邻接走廊的概率为0.25

"""
X_INDEX = 0
Y_INDEX = 1
L_INDEX = 2
W_INDEX = 3
SPACE_ID_INDEX = 4
TYPE_INDEX = 5
B1_INDEX = 6
B2_INDEX = 7
B3_INDEX = 8
B4_INDEX = 9

