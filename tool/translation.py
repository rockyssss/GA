from dao import rectangle_index as rec
from numba import jit


# @jit
def translation(rectangles_array, delta_x, delta_y):
    """
    rectangle的平移：传入rectangles_array,和要对应平移的rectangle的索引,平移的距离delta_x, delta_y
    :param rectangles_array:
    :param index:
    :param delta_x:
    :param delta_y:
    :return:
    """
    # X对应的数组下标
    rectangles_array[rec.X_MIN_INDEX] = rectangles_array[rec.X_MIN_INDEX] + delta_x
    # Y对应的数组下标
    rectangles_array[rec.Y_MIN_INDEX] = rectangles_array[rec.Y_MIN_INDEX] + delta_y
    return rectangles_array


# space的平移
def tanslationrec(spacesarray, spacesdic, spaceid, delta_x, delta_y):
    listrec = spacesdic.get(spaceid)
    for index in listrec:
        spacesarray[index][rec.X_MIN_INDEX] = spacesarray[index][rec.X_MIN_INDEX] + delta_x
        # Y对应的数组下标
        spacesarray[index][rec.Y_MIN_INDEX] = spacesarray[index][rec.Y_MIN_INDEX] + delta_y
    return spacesarray