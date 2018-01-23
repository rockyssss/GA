# _*_ coding: UTF-8 _*_
import numpy as np
from dao import spaceApoint as sap


def dao2np(spaces):
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
    :param spaces:
    :return:
    """

    recs = np.zeros((len(spaces), 10))
    for index, space in enumerate(spaces):
        rec = recs[index]
        # ids.append(index)
        rec[4] = index

        # types.append(float(space.type))
        rec[5] = float(space.type)
        for rectangle in space.rectangle:
            # ls.append(rectangle.length)
            rec[2] = rectangle.length
            # ws.append(rectangle.width)
            rec[3] = rectangle.width
            # b1s.append(rectangle.b1)
            rec[6] = rectangle.b1
            # b2s.append(rectangle.b2)
            rec[7] = rectangle.b2
            # b3s.append(rectangle.b3)
            rec[8] = rectangle.b3
            # b4s.append(rectangle.b4)
            rec[9] = rectangle.b4
            point3d = rectangle.point3d
            # xs.append(point3d.x)
            rec[0] = point3d.x
            # ys.append(point3d.y)
            rec[1] = point3d.y

    return recs


def np2dao(recs):
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
        :param spaces:
        :return:
        """
    direction=sap.Direction(1,0,1,1,0)
    spaces = []
    for rec in recs:
        p3d=sap.Point3D(rec[0], rec[1], 0)
        rectan=sap.Rectangle(rec[2],rec[3],p3d,direction,rec[1],rec[2],rec[3],rec[4])
        space=sap.Space("",rec[4],rectan)
        spaces.append(space)
    return spaces
