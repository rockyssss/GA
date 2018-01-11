# _*_ coding: UTF-8 _*_

from dao import spaceApoint as sap
from tool import other

"""
    批量房间更改,在房间数据都有的情况下
"""


def chagePoints(spaces, beginPoint, direction):
    for space in spaces:
        space.rectangle[0].point3d = beginPoint
        dxy = other.crossSingle(float(beginPoint.x), float(beginPoint.y),
                                [space.rectangle[0].length, space.rectangle[0].width], direction)
        beginPoint = sap.Point3D(dxy[0], dxy[1], 0)
