# _*_ coding: UTF-8 _*_
import numpy as np

def isnumber(a_string):
    """
    判断字符串是否可以转为float型
    :param a_string:
    :return:
    """
    try:
        float(a_string)
        return True
    except:
        return False


def new_ros(X_MIN, Y_MIN, Xm, Ym, L, W, SPACE_ID, TYPE, B1, B2, B3, B4, BOUND):
    """
    新建rectangle或者space，返回新建的np.array(传入13个参数，分别为矩形(空间)中：最小X，最小的Y，最大X，最大的Y，长度L，宽度W，space_id，type，边界B1，B2，B3，B4，区域边界索引BOUND)
    :param X_MIN: 矩形a 左下角点x坐标(最小的x)为X
    :param Y_MIN: 矩形a 左下角点y坐标(最小的y)为Y
    :param Xm: 矩形a 左下角点x坐标(最大的x)为X_MAX
    :param Ym: 矩形a 左下角点y坐标(最大的y)为Y_MAX
    :param L: 矩形a 长度为L
    :param W: 矩形a 宽度为W
    :param SPACE_ID: 矩形a 对应的space_id为SPACE_ID
    :param TYPE: 矩形a 对应的type为TYPE
    :param B1: 矩形a 若B1>0则B1边邻接的房间为B1号房间，否则若B1<0则B1边邻接走廊的概率为B1
    :param B2: 矩形a 若B2>0则B1边邻接的房间为B1号房间，否则若B1<0则B1边邻接走廊的概率为B2
    :param B3: 矩形a 若B3>0则B1边邻接的房间为B1号房间，否则若B1<0则B1边邻接走廊的概率为B3
    :param B4: 矩形a 若B4>0则B1边邻接的房间为B1号房间，否则若B1<0则B1边邻接走廊的概率为B4
    :param BOUND: 矩形a 所在的区域边界的索引为BOUND
    :return:
    """
    return np.array([X_MIN, Y_MIN, Xm, Ym, L, W, SPACE_ID, TYPE, B1, B2, B3, B4, BOUND])


