# _*_ coding: UTF-8 _*_

from dao import spaceApoint as sap


# 给走量定义矩形属性
def getCorridor(spaces, Clength, Cwidth, direction):
    # lineroom
    beginRoomPoi = spaces[0].rectangle[0].point3d
    endRoomPoi = spaces[len(spaces) - 1].rectangle[0].point3d
    length = 0
    if (direction.dx != 0):
        length = abs(endRoomPoi.x - beginRoomPoi.x) + spaces[len(spaces) - 1].rectangle[0].length
        x = spaces[0].rectangle[0].point3d.x
        y = spaces[0].rectangle[0].point3d.y + spaces[len(spaces) - 1].rectangle[0].width * direction.y
    else:
        length = abs(endRoomPoi.y - beginRoomPoi.y) + spaces[len(spaces) - 1].rectangle[0].length
        x = spaces[0].rectangle[0].point3d.x + spaces[len(spaces) - 1].rectangle[0].width * direction.x
        y = spaces[0].rectangle[0].point3d.y
    # corridor

    if Clength != 0:
        return sap.Rectangle(Clength, Cwidth, sap.Point3D(x, y, 0), direction)

    return sap.Rectangle(length, Cwidth, sap.Point3D(x, y, 0), direction)
    # rectangle=sap.Rectangle
