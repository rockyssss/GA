# _*_ coding: UTF-8 _*_
# import numpy as np
import pylab as plt
import matplotlib.patches as patches

"""
    根据张学芳代码修改
"""

def draw(spaces,direction):
    fig = plt.figure()
    # plt.xlim(-5, 100)
    # plt.ylim(-5, 100)
    floor1 = fig.add_subplot(1, 1, 1)
    for space in spaces:
        x = []  # Make an array of x values
        y = []  # Make an array of y values for each x value
        rectangle = space.rectangle[0]
        name=space.name
        type=space.type
        length = rectangle.length
        width = rectangle.width
        point = rectangle.point3d
        if (direction.dx != 0):
            x0 = point.x
            y0 = point.y
            x.append(x0)
            y.append(y0)
            x1 = point.x + length * direction.dx
            y1 = point.y
            x.append(x1)
            y.append(y1)
            x2 = point.x + length * direction.dx
            y2 = point.y + width * direction.y
            x.append(x2)
            y.append(y2)
            x3 = point.x
            y3 = point.y + width * direction.y
            x.append(x3)
            y.append(y3)
            # 回到原点闭合成矩形
            x.append(x0)
            y.append(y0)
            center_pointx = point.x + length * direction.dx / 2
            center_pointy = point.y + width * direction.y / 2
        else:
            x0 = point.x
            y0 = point.y
            x.append(x0)
            y.append(y0)
            x1 = point.x
            y1 = point.y + length * direction.dy
            x.append(x1)
            y.append(y1)
            x2 = point.x + width * direction.x
            y2 = point.y + length * direction.dy
            x.append(x2)
            y.append(y2)
            x3 = point.x + width * direction.x
            y3 = point.y
            x.append(x3)
            y.append(y3)
            # 回到原点闭合成矩形
            x.append(x0)
            y.append(y0)
            center_pointx = point.x + width * direction.x / 2
            center_pointy = point.y + length * direction.dy / 2
        floor1.text(center_pointx, center_pointy, name, fontproperties='SimHei', horizontalalignment='center',
                    verticalalignment='center',)
        #这按照房间plot
        floor1.plot(x, y, '-')
        # corner_points = np.vstack((corner_points, corner_points.x))
        # floor1.plot(corner_points[:, 0], corner_points[:, 1], 'v:')
    plt.figure(figsize=(1, 1))
    plt.show()  # show the plot on the screen


def GA_draw_data(spaces):
    plt.cla()
    for space in spaces:
        x = []  # Make an array of x values
        y = []  # Make an array of y values for each x value
        rectangle = space.rectangle[0]
        direction = space.rectangle[0].Direction
        name = space.name
        type = space.type
        length = rectangle.length
        width = rectangle.width
        point = rectangle.point3d
        if direction.dx != 0:
            x0 = point.x
            y0 = point.y
            x.append(x0)
            y.append(y0)
            x1 = point.x + length * direction.dx
            y1 = point.y
            x.append(x1)
            y.append(y1)
            x2 = point.x + length * direction.dx
            y2 = point.y + width * direction.y
            x.append(x2)
            y.append(y2)
            x3 = point.x
            y3 = point.y + width * direction.y
            x.append(x3)
            y.append(y3)
            # 回到原点闭合成矩形
            x.append(x0)
            y.append(y0)
            center_pointx = point.x + length * direction.dx / 2
            center_pointy = point.y + width * direction.y / 2
        else:
            x0 = point.x
            y0 = point.y
            x.append(x0)
            y.append(y0)
            x1 = point.x
            y1 = point.y + length * direction.dy
            x.append(x1)
            y.append(y1)
            x2 = point.x + width * direction.x
            y2 = point.y + length * direction.dy
            x.append(x2)
            y.append(y2)
            x3 = point.x + width * direction.x
            y3 = point.y
            x.append(x3)
            y.append(y3)
            # 回到原点闭合成矩形
            x.append(x0)
            y.append(y0)
            center_pointx = point.x + width * direction.x / 2
            center_pointy = point.y + length * direction.dy / 2
        plt.text(center_pointx, center_pointy, name, fontproperties='SimHei', horizontalalignment='center',
                    verticalalignment='center', )
        # 这按照房间plot
        plt.plot(x, y, '-')


def draw_update(floor1, spaces, direction):
    for space in spaces:
        x = []  # Make an array of x values
        y = []  # Make an array of y values for each x value
        rectangle = space.rectangle[0]
        name = space.name
        type = space.type
        length = rectangle.length
        width = rectangle.width
        point = rectangle.point3d
        if (direction.dx != 0):
            x0 = point.x
            y0 = point.y
            x.append(x0)
            y.append(y0)
            x1 = point.x + length * direction.dx
            y1 = point.y
            x.append(x1)
            y.append(y1)
            x2 = point.x + length * direction.dx
            y2 = point.y + width * direction.y
            x.append(x2)
            y.append(y2)
            x3 = point.x
            y3 = point.y + width * direction.y
            x.append(x3)
            y.append(y3)
            # 回到原点闭合成矩形
            x.append(x0)
            y.append(y0)
            center_pointx = point.x + length * direction.dx / 2
            center_pointy = point.y + width * direction.y / 2
        else:
            x0 = point.x
            y0 = point.y
            x.append(x0)
            y.append(y0)
            x1 = point.x
            y1 = point.y + length * direction.dy
            x.append(x1)
            y.append(y1)
            x2 = point.x + width * direction.x
            y2 = point.y + length * direction.dy
            x.append(x2)
            y.append(y2)
            x3 = point.x + width * direction.x
            y3 = point.y
            x.append(x3)
            y.append(y3)
            # 回到原点闭合成矩形
            x.append(x0)
            y.append(y0)
            center_pointx = point.x + width * direction.x / 2
            center_pointy = point.y + length * direction.dy / 2
        floor1.text(center_pointx, center_pointy, name, fontproperties='SimHei', horizontalalignment='center',
                    verticalalignment='center', )
        # 这按照房间plot
        floor1.plot(x, y, '-')
        # corner_points = np.vstack((corner_points, corner_points.x))
        # floor1.plot(corner_points[:, 0], corner_points[:, 1], 'v:')
        # plt.figure(figsize=(1, 1))
        # plt.show()  # show the plot on the screen
