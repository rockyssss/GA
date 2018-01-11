# _*_ coding: UTF-8 _*_

"""
    编写等宽矩形行
"""


# 平均宽度
def averageWidth(spaces):
    sum = 0
    for space in spaces:
        width = space.rectangle[0].width
        sum += width
    aequilate = sum / len(spaces)
    return aequilate


# 更新space的宽度和长度,长=长*宽/等宽,坐标也要更新
# 输入:房间总信息,平均宽
def lenPoinAftAequilate(spaces, aequilate):
    for space in spaces:
        space.rectangle[0].length = space.rectangle[0].length * space.rectangle[0].width / aequilate
        space.rectangle[0].width = aequilate
