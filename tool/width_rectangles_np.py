# _*_ coding: UTF-8 _*_
import copy
import numpy as np
import dao.rectangle_index as ri

"""
    编写等宽矩形行
"""


# 平均宽度
def averageWidth_np(rectangles):
    best_width_np(rectangles)
    sum = 0
    for rectangle in rectangles:
        width = rectangles[ri.W_INDEX]
        sum += width
    aequilate = sum / len(rectangles)
    return aequilate





# 更新space的宽度和长度,长=长*宽/等宽,坐标也要更新
# 输入:房间总信息,平均宽
def lenPoinAftAequilate(rectangles, aequilate):
    for rectangle in rectangles:
        rectangle[ri.L_INDEX] =rectangle[ri.L_INDEX] * rectangle[ri.W_INDEX] / aequilate
        rectangle[ri.W_INDEX] = aequilate





def best_width_np(rectangles):
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
    widths = []
    lengths = []
    for rectangle in rectangles:
        widths.append(rectangle[ri.W_INDEX])
        lengths.append(rectangle[ri.L_INDEX])

    # 取得长宽的偏差
    # bw bl存储最佳长宽
    pw = distortion(widths)
    pl = distortion(lengths)
    if pw > pl:
        bw = copy.deepcopy(lengths)
        bl = copy.deepcopy(widths)
        best = pl
    else:
        bw = copy.deepcopy(widths)
        bl = copy.deepcopy(lengths)
        best = pw

    # 长宽颠倒看效果
    for index in range(len(widths)):






        # 下标为index的长宽互换
        # print("互换")
        # print(widths[i])
        temp = widths[index]
        widths[index] = lengths[index]
        lengths[index] = temp
        # 新的pw,pl
        pw = distortion(widths)
        pl = distortion(lengths)
        # 几种情况处理逻辑,保证每次循环完毕后,pw和widths是同步的
        if pw <= pl:
            # 宽整体不变情况下,局部变化
            if pw < best:
                temp = bw[index]
                bw[index] = bl[index]
                bl[index] = temp
                best = pw
            else: #变回来
                temp = widths[index]
                widths[index] = lengths[index]
                lengths[index] = temp
        else:   #长<宽,
            #长宽互换,并且局部变化,无分先后
            #互换的包括lengths 和widths和bw bl
            if pl<best:
                # 局部变化
                temp = bw[index]
                bw[index] = bl[index]
                bl[index] = temp
                best = pw

                temp = widths[index]
                widths[index] = lengths[index]
                lengths[index] = temp

                # 全局变化
                temWidth = lengths
                temLength = widths
                widths = temWidth
                lengths = temLength

                temWidth = bl
                temLength = bw
                bw = temWidth
                bl = temLength
            else: #变回来
                temp = widths[index]
                widths[index] = lengths[index]
                lengths[index] = temp


    # 得到最佳长宽后,更新spaces
    for i, rec in enumerate(rectangles):
        rectangle[ri.W_INDEX]= bw[i]
        rectangle[ri.L_INDEX]= bl[i]




# 求list的偏差值
def distortion(items):
    """
    先求平均值,然后每个元素和平均值相减少
    :param items:
    :return:
    """
    sum = 0
    for item in items:
        sum += item
    avg = sum / len(items)

    # 求偏差值
    distor = 0
    for item in items:
        distor += abs(item - avg)
    return distor
