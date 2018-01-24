# _*_ coding: UTF-8 _*_
import numpy as np
from dao import spaceApoint as sap
import dao.rectangle_index as ri

def dao2np(spaces):
    """
    RS_LEN=13
    X_MIN_INDEX = 0
    Y_MIN_INDEX = 1
    X_MAX_INDEX = 2
    Y_MAX_INDEX = 3
    L_INDEX = 4
    W_INDEX = 5
    SPACE_ID_INDEX = 6
    TYPE_INDEX = 7
    B1_INDEX = 8
    B2_INDEX = 9
    B3_INDEX = 10
    B4_INDEX = 11
    BOUND_INDEX = 12
    :param spaces:
    :return:
    """

    recs = np.zeros((len(spaces), ri.RS_LEN))
    for index, space in enumerate(spaces):
        rec = recs[index]
        # ids.append(index)
        rec[ri.SPACE_ID_INDEX] = index

        # types.append(float(space.type))
        rec[ri.TYPE_INDEX] = float(space.type)
        for rectangle in space.rectangle:
            # ls.append(rectangle.length)
            rec[ri.L_INDEX] = rectangle.length
            # ws.append(rectangle.width)
            rec[ri.W_INDEX] = rectangle.width
            # b1s.append(rectangle.b1)
            rec[ri.B1_INDEX] = rectangle.b1
            # b2s.append(rectangle.b2)
            rec[ri.B2_INDEX] = rectangle.b2
            # b3s.append(rectangle.b3)
            rec[ri.B3_INDEX] = rectangle.b3
            # b4s.append(rectangle.b4)
            rec[ri.B4_INDEX] = rectangle.b4
            point3d = rectangle.point3d
            # xs.append(point3d.x)
            rec[ri.X_MIN_INDEX] = point3d.x
            # ys.append(point3d.y)
            rec[ri.Y_MIN_INDEX] = point3d.y

    return recs


def np2dao(recs):
    """
        X_INDEX = 0
        Y_MIN_INDEX = 1
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
        p3d=sap.Point3D(rec[ri.X_MIN_INDEX], rec[ri.Y_MIN_INDEX], 0)
        rectan=sap.Rectangle(rec[ri.L_INDEX],rec[ri.W_INDEX],p3d,direction,rec[ri.Y_MIN_INDEX],rec[ri.L_INDEX],rec[ri.W_INDEX],rec[ri.SPACE_ID_INDEX])
        space=sap.Space("",rec[ri.SPACE_ID_INDEX],rectan)
        spaces.append(space)
    return spaces
