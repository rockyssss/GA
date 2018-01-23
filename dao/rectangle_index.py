# _*_ coding: UTF-8 _*_
"""
示例:

a = np.array([0, 1, 2, 7, 8, 3, 4, 5, 2, -0.25, -0.5, -0.25])
若a = np.array([X, Y, Xm, Ym, L, W, SPACE_ID, TYPE, B1, B2, B3, B4])
此时则有：
a[X_INDEX]为X----------------------矩形a 左下角点x坐标为X
a[Y_INDEX]为Y----------------------矩形a 左下角点y坐标为Y
a[X_INDEX]为Xm---------------------矩形a 左下角点x坐标为Xm
a[Y_INDEX]为Ym---------------------矩形a 左下角点y坐标为Ym
a[L_INDEX]为L----------------------矩形a 长度为L
a[W_INDEX]为W----------------------矩形a 宽度为W
a[SPACE_ID_INDEX]为SPACE_ID--------矩形a 对应的space_id为SPACE_ID
a[TYPE_INDEX]为TYPE----------------矩形a 对应的type为TYPE
a[B1_INDEX]为B1--------------------矩形a 若B1>0则B1边邻接的房间为B1号房间，否则若B1<0则B1边邻接走廊的概率为B1
a[B2_INDEX]为B2--------------------矩形a 若B2>0则B1边邻接的房间为B1号房间，否则若B1<0则B1边邻接走廊的概率为B2
a[B3_INDEX]为B3--------------------矩形a 若B3>0则B1边邻接的房间为B1号房间，否则若B1<0则B1边邻接走廊的概率为B3
a[B4_INDEX]为B4--------------------矩形a 若B4>0则B1边邻接的房间为B1号房间，否则若B1<0则B1边邻接走廊的概率为B4

"""
X_INDEX = 0
Y_INDEX = 1
Xm_INDEX = 2
Ym_INDEX = 3
L_INDEX = 4
W_INDEX = 5
SPACE_ID_INDEX = 6
TYPE_INDEX = 7
B1_INDEX = 8
B2_INDEX = 9
B3_INDEX = 10
B4_INDEX = 11

