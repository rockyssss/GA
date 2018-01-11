# _*_ coding: UTF-8 _*_

import copy
"""
    通过方向,判断在beginpoint在哪里,然后将其转换到左上角

"""


def judge(spaces, direction):
    if (direction.x == 1):
        if (direction.y == 1):
            if (direction.dx == 1):
                go2topleftcorner1(spaces, direction)
            else:  # dy==1
                go2topleftcorner2(spaces, direction)
        else:  # y==-1
            if (direction.dx == 1):
                go2topleftcorner7(spaces, direction)
            else:  # dy==1
                go2topleftcorner8(spaces, direction)

    else:  # x==-1
        if (direction.y == 1):
            if (direction.dx == -1):
                go2topleftcorner3(spaces, direction)
            else:
                go2topleftcorner4(spaces, direction)
        else:  # direction.y==-1
            if (direction.dx == -1):
                go2topleftcorner5(spaces, direction)
            else:
                go2topleftcorner6(spaces, direction)


# 第一象限    dx==1
def go2topleftcorner1(spaces, direction):
    # for space in spaces:
    #     rectangle = space.rectangle
    #     point3d = rectangle.point3d
    #     point3d.y = point3d.y + rectangle.width * direction.y
    pass


# 第一象限    dy==1
def go2topleftcorner2(spaces, direction):
    for space in spaces:
        rectangle = space.rectangle[0]
        point3d = rectangle.point3d
        # point3d.y = point3d.y + rectangle.length * direction.y
        # 修改长宽
        length = copy.deepcopy(rectangle.width)
        width = copy.deepcopy(rectangle.length)
        rectangle.length = length
        rectangle.width = width
        rectangle.Direction = (1.0, 0.0, 1, 1, 0)


# 第二象限    dx==-1
def go2topleftcorner3(spaces, direction):
    for space in spaces:
        rectangle = space.rectangle[0]
        point3d = rectangle.point3d
        point3d.x = point3d.x + rectangle.length * direction.x
        rectangle.Direction = (1.0, 0.0, 1, 1, 0)


# 第二象限    dy==1
def go2topleftcorner4(spaces, direction):
    for space in spaces:
        rectangle = space.rectangle[0]
        point3d = rectangle.point3d
        point3d.x = point3d.x + rectangle.width * direction.x
        # 修改长宽
        length = copy.deepcopy(rectangle.width)
        width = copy.deepcopy(rectangle.length)
        rectangle.length = length
        rectangle.width = width
        rectangle.Direction = (1.0, 0.0, 1, 1, 0)


# 第三象限    dx==-1
def go2topleftcorner5(spaces, direction):
    for space in spaces:
        rectangle = space.rectangle[0]
        point3d = rectangle.point3d
        point3d.x = point3d.x + rectangle.length * direction.x
        point3d.y = point3d.y + rectangle.width * direction.y
        rectangle.Direction = (1.0, 0.0, 1, 1, 0)

# 第三象限    dy==-1
def go2topleftcorner6(spaces, direction):
    for space in spaces:
        rectangle = space.rectangle[0]
        point3d = rectangle.point3d
        point3d.x = point3d.x + rectangle.width * direction.x
        point3d.y = point3d.y + rectangle.length * direction.y
        # 修改长宽
        length = copy.deepcopy(rectangle.width)
        width = copy.deepcopy(rectangle.length)
        rectangle.length = length
        rectangle.width = width
        rectangle.Direction = (1.0, 0.0, 1, 1, 0)


# 第四象限   dx==1
def go2topleftcorner7(spaces, direction):
    for space in spaces:
        rectangle = space.rectangle[0]
        point3d = rectangle.point3d
        point3d.y = point3d.y + rectangle.width * direction.y
        rectangle.Direction = (1.0, 0.0, 1, 1, 0)


# 第四象限
def go2topleftcorner8(spaces, direction):
    for space in spaces:
        rectangle = space.rectangle[0]
        point3d = rectangle.point3d
        point3d.y = point3d.y + rectangle.length * direction.y

        # 修改长宽
        length = copy.deepcopy(rectangle.width)
        width = copy.deepcopy(rectangle.length)
        rectangle.length = length
        rectangle.width = width
        rectangle.Direction = (1.0, 0.0, 1, 1, 0)
