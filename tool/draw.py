# _*_ coding: UTF-8 _*_
import numpy as np
import pylab as plt
import dao.rectangle_index as ri



def draw(rectanges):
    """
    x_index = 0
    y_index = 1
    l_index = 2
    w_index = 3
    space_id_index = 4
    type_index = 5
    b1_index = 6
    b2_index = 7
    b3_index = 8
    b4_index = 9

    :param rectanges:
    :return:
    """
    
    fig = plt.figure()
    # plt.xlim(-5, 100)
    # plt.ylim(-5, 100)
    floor1 = fig.add_subplot(1, 1, 1)
    
    for rec in rectanges:
        x = []  # Make an array of x values
        y = []  # Make an array of y values for each x value


        x0 = rec[0]
        y0 = rec[1]
        x.append(x0)
        y.append(y0)
        x1 = rec[0]
        y1 = rec[1] + rec[3]
        x.append(x1)
        y.append(y1)
        x2 = rec[0] + rec[2]
        y2 = rec[1] + rec[3]
        x.append(x2)
        y.append(y2)
        x3 = rec[0] + rec[2]
        y3 = rec[1]
        x.append(x3)
        y.append(y3)
        # 回到原点闭合成矩形
        x.append(x0)
        y.append(y0)
        center_pointx = rec[0] + rec[2]  / 2
        center_pointy = rec[1] + rec[3] / 2
        floor1.text(center_pointx, center_pointy, ri.Adict[rec[5]], fontproperties='SimHei', horizontalalignment='center',
                    verticalalignment='center' ,)
        # 这按照房间plot
        floor1.plot(x, y, '-')
        # corner_points = np.vstack((corner_points, corner_points.x))
        # floor1.plot(corner_points[:, 0], corner_points[:, 1], 'v:')
    plt.figure(figsize=(1, 1))
    plt.show()  # show the plot on the screen