# _*_ coding: UTF-8 _*_
import numpy as np
import pylab as plt
import dao.rectangle_index as ri



def draw(rectanges):
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

    :param rectanges:
    :return:
    """

    plt.cla()
    for rec in rectanges:
        x = []  # Make an array of x values
        y = []  # Make an array of y values for each x value

        x0 = rec[ri.X_MIN_INDEX]
        y0 = rec[ri.Y_MIN_INDEX]
        x.append(x0)
        y.append(y0)
        x1 = rec[ri.X_MIN_INDEX]
        y1 = rec[ri.Y_MIN_INDEX] + rec[ri.W_INDEX]
        x.append(x1)
        y.append(y1)
        x2 = rec[ri.X_MIN_INDEX] + rec[ri.L_INDEX]
        y2 = rec[ri.Y_MIN_INDEX] + rec[ri.W_INDEX]
        x.append(x2)
        y.append(y2)
        x3 = rec[ri.X_MIN_INDEX] + rec[ri.L_INDEX]
        y3 = rec[ri.Y_MIN_INDEX]
        x.append(x3)
        y.append(y3)
        # 回到原点闭合成矩形
        x.append(x0)
        y.append(y0)
        center_pointx = rec[ri.X_MIN_INDEX] + rec[ri.L_INDEX] / 2
        center_pointy = rec[ri.Y_MIN_INDEX] + rec[ri.W_INDEX] / 2
        plt.text(center_pointx, center_pointy, ri.TYPE_NAME_DICT[int(rec[ri.TYPE_INDEX])], fontproperties='SimHei', horizontalalignment='center',
                    verticalalignment='center',)
        # 这按照房间plot
        plt.plot(x, y, '-')
