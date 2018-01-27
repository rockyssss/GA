# _*_ coding: UTF-8 _*_
from dao import rectangle_index as ri


def dx(k1, k2):
    """
    两个矩形之间在x轴距离
    :param k1: rectange1
    :param k2:
    :return:
    """
    # 获取rectangle的对应轴2个坐标值
    rk1 = [k1[ri.X_MIN_INDEX], k1[ri.X_MIN_INDEX] + k1[ri.L_INDEX]]
    rk2 = [k2[ri.X_MIN_INDEX], k2[ri.X_MIN_INDEX] + k2[ri.L_INDEX]]

    # max{Vx(K1),Vx(K2)}
    b_max = max(max(rk1), max(rk2))
    b_min = min(min(rk1), min(rk2))

    result = b_max - b_min - k1[ri.L_INDEX] - k2[ri.L_INDEX]
    return result


def dy(k1, k2):
    """
    两个矩形之间在x轴距离
    :param k1: rectange1
    :param k2:
    :return:
    """
    # 获取rectangle的对应轴2个坐标值
    rk1 = [k1[ri.Y_MIN_INDEX], k1[ri.Y_MIN_INDEX] + k1[ri.W_INDEX]]
    rk2 = [k2[ri.Y_MIN_INDEX], k2[ri.Y_MIN_INDEX] + k2[ri.W_INDEX]]

    # max{Vx(K1),Vx(K2)}
    b_max = max(max(rk1), max(rk2))
    b_min = min(min(rk1), min(rk2))

    result = b_max - b_min - k1[ri.W_INDEX] - k2[ri.W_INDEX]
    return result


def distance(k1, k2, c=0.2):
    if dx(k1, k2) >= 0 and dy(k1, k2) >= 0:
        return dx(k1, k2) + dy(k1, k2) + c
    elif dx(k1, k2) >= 0 and dy(k1, k2) + c >= 0:
        return dx(k1, k2) + dy(k1, k2) + c
    elif dx(k1, k2) + c >= 0 and dy(k1, k2) >= 0:
        return dx(k1, k2) + dy(k1, k2) + c
    # elif dx(k1, k2) >= 0 and dy(k1, k2) + c < 0:
    elif dy(k1, k2) + c < 0 <= dx(k1, k2):
        return dx(k1, k2)
    # elif dx(k1, k2) + c < 0 and dy(k1, k2) >= 0:
    elif dx(k1, k2) + c < 0 <= dy(k1, k2):
        return dy(k1, k2)
    elif dx(k1, k2) + c >= 0 and dy(k1, k2) + c >= 0:
        a = min(dx(k1, k2) + c, dy(k1, k2) + c)
        b = max(dx(k1, k2), dy(k1, k2))
        return a - b
    # elif dx(k1, k2) + c >= 0 and dy(k1, k2) + c  < 0:
    elif dy(k1, k2) + c < 0 <= dx(k1, k2) + c:
        return abs(dx(k1, k2))
    elif dx(k1, k2) + c < 0 <= dy(k1, k2) + c:
        return abs(dy(k1, k2))
    elif dx(k1, k2) + c < 0 and dy(k1, k2) + c < 0:
        return min(abs(dx(k1, k2)), abs(dy(k1, k2)))
