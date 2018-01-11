#!/usr/bin/python
# _*_ coding: UTF-8 _*_
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


import os

from dao import spaceApoint as sap
from planinitrunner import plot_draw, get_cpd
from tool import csvBrief, other, widthRectangles, spacesChanges, corridor, Direction2Point
from tool import lengthAndWidth as lw
from planinitrunner import AreaAndNumber
import copy
import math

def linear(total_area):
    result1 = AreaAndNumber.LinearCoef_1()
    result2 = AreaAndNumber.LinearCoef_2()
    result4 = AreaAndNumber.test_negative_incline()
    result6 = AreaAndNumber.test_negative_incline_1()
    ll = AreaAndNumber.RoundNumbers(result1, result4)
    result5 = AreaAndNumber.RoundNumbers(result2, result6)
    RoomNumber = AreaAndNumber.Calculate(AreaAndNumber.WeightArea(ll, total_area), total_area)
    OperationOne = RoomNumber[4] / 20 * 3
    OprationTwo = RoomNumber[4] * 1 / 20
    OperationTree = RoomNumber[4] * 10 / 20
    OperationFour = RoomNumber[4] * 4 / 20
    OpreationFive = RoomNumber[4] * 2 / 20
    RoomList = AreaAndNumber.OperationRoom(RoomNumber[4])
    AreaAndNumber.Number.extend(RoomList)
    ResultArea = AreaAndNumber.Calculate(AreaAndNumber.Weight(result5, math.floor(RoomNumber[4] / 50)),
                                         math.floor(RoomNumber[4] / 50))
    ResultArea.append(15)
    ResultArea.append(10)
    ResultArea.append(10)
    ResultArea.append(20)
    ResultArea.append(20)
    ResultArea.append(50)
    ResultArea.append(15)
    ResultArea.append(OperationOne)
    ResultArea.append(OprationTwo)
    ResultArea.append(OperationTree)
    ResultArea.append(OperationFour)
    ResultArea.append(OpreationFive)
    AreaAndNumber.Number[32] = AreaAndNumber.math.floor(ResultArea[32] / 8.5)
    AreaAndNumber.Number[33] = AreaAndNumber.math.floor(ResultArea[33] / 25)
    AreaAndNumber.Number[36] = AreaAndNumber.math.floor(ResultArea[36] / 2)
    ResultArea[42] = ResultArea[42] * AreaAndNumber.Number[42]
    ResultArea[44] = AreaAndNumber.Number[44] * ResultArea[44]
    res = AreaAndNumber.ClassArea(total_area, ResultArea)
    # 个数
    # print(AreaAndNumber.Number)
    # 面积
    # print(AreaAndNumber.ClassArea(3000, ResultArea))
    area = AreaAndNumber.ClassArea(total_area, ResultArea)

    # i = 0
    # while i < len(area) - 12:
    #     area[i] /= 3
    #     i += 1
    # print(area)
    return AreaAndNumber.Number, area


# # 测试全流程
# ab = os.getcwd()
# dataPath2 = os.getcwd() + "/../file/ratio.csv"  # 长宽比数据
# dict2 = csvBrief.readListCSV(dataPath2)
# data2 = dict2[1:]  # 长宽数据集合
#
# Aname = dict2[0]  # 数据的列名称集合
# Anum = AreaAndNumber.Number
# Aarea = AreaAndNumber.ClassArea(3000, ResultArea)
# # Alength=data2[0]
# # Awidth=data2[1]
# Alength = data2[0]
# Awidth = data2[1]
# Atype = data2[3]
#
# p = {"51443315": 1}
# cpdPath = "\\..\\file\\cyk\\cyk.txt"
#
# dict = {}
# for id in range(len(Aname)):
#     dict[Atype[id]] = id


def ruffle(Aname, Atype, p, cpdPath, Anum):
    # dict {'51441115': 0, '51443500': 1, '51442113': 2, '51172100': 3, '51171300': 4, '51348400': 5,....
    dict = {}
    for id in range(len(Aname)):
        dict[Atype[id]] = id
    # 写一个dict用来解决临接关系的顺序问题

    # p = {'51442115': 1}
    # dict = {'51578800': 0, '51172100': 1, '51445500': 2, '51443315': 3, '51442115': 4, '51442113': 5, '51543115': 6}
    # list1 = [2, 2, 3, 1, 2, 2, 1]

    cpd = get_cpd.Cpd(cpdPath)
    # lists  房间编号:对应的临接概率, 最后一位是你给它的房间编号集合
    lists = get_cpd.get_cpd(p, cpd)
    # values:<class 'list'>: ['65131300', '65131300', '51444500', '51444500', '65131100', '65131100', '51178400', '51178400', '51443911', '51348000', '51172382', '51178600', '51178000']
    values = get_cpd.loop(p, dict, lists, Anum, cpdPath)
    return values


def get_bic_data(Aname, Anum, Aarea, Alength, Awidth, values, dict):
    """
    从整体数据中,选出要排列的数据
    :param Aname:
    :param Anum:
    :param Aarea:
    :param Alength:
    :param Awidth:
    :param Atype:这个不需要,不传入了
    :param values:  房间排布的type数据['65131300', '65131300', '51444500',
    :param dict: type:对应下标  {'51441115': 0, '51443500': 1, '51442113': 2, '51172100': 3,
    :return:
    """

    medicName = []
    medicType = values
    medicArea = []
    length = []
    width = []
    for type in values:
        # 获取下标
        num = dict.get(type)
        # 拿取名称集合
        name = Aname[num]
        medicName.append(name)
        # 拿取个数集合,从而求单位面积
        medicNum = Anum[num]
        # 拿取面积集合
        medicArea.append(float(Aarea[num]) / float(medicNum))
        # 手书长宽比例数据
        length.append(float(Alength[num]))
        width.append(float(Awidth[num]))
    return {"medicName": medicName, "medicType": medicType, "medicArea": medicArea, "length": length, "width": width}


# 初始化一排房间
def rowSpaces(spaces, direction, linePoint, medicName, medicType, medicArea, length, width):
    # 得到真实的长宽
    listLW = []
    for i in range(len(medicName)):
        tmp = lw.getm2n(float(medicArea[i]), float(length[i]), float(width[i]))
        listLW.append(tmp)

    # 房间数据开始存储,第一次初始化
    # spaces = []
    beginPoint = copy.deepcopy(linePoint)
    for i in range(len(medicName)):
        rectangle = []
        rectangle.append(sap.Rectangle(listLW[i][0], listLW[i][1], beginPoint, direction))
        space = sap.Space(medicName[i], medicType[i], rectangle)
        spaces.append(space)
        # 这里根据你要填入是是列还是行更新beginPoint
        # 存储完成后,更新行代码
        dxy = other.crossSingle(beginPoint.x, beginPoint.y, listLW[i], direction)
        beginPoint = sap.Point3D(dxy[0], dxy[1], 0)


        # # 所有房间都存储后,进行取值绘图
        # paintSpaces2.draw(spaces, direction)


# 每排修改成等宽后的代码
def aequilate(spaces, linePoint, direction):
    # 求等宽值aequilate
    aequilate = widthRectangles.averageWidth(spaces)
    # 等宽更新长度
    widthRectangles.lenPoinAftAequilate(spaces, aequilate)
    # 更新坐标
    beginPoint = copy.deepcopy(linePoint)
    spacesChanges.chagePoints(spaces, beginPoint, direction)

    # paintSpaces2.draw(spaces, direction)


"""
    画一个走廊,走廊必定依附于旁边成排的房间,他的一个长度就是
"""


def addCorridor(spaces, direction, Clength, Cwidth, num, Aname, type):
    # 定义走廊宽度
    rectangle = [corridor.getCorridor(spaces, Clength, Cwidth, direction)]
    # 34清洁走廊
    space = sap.Space(Aname[num], type[num], rectangle)
    spaces.append(space)


def addOneSpace(spaces, beginPoint, Clength, Cwidth, name, type, direction):
    """

    :param beginPoint:
    :param Clength:
    :param Cwidth:
    :param name: name自己注入
    :return:
    """
    rectangle = [sap.Rectangle(Clength, Cwidth, beginPoint, direction)]
    # 34清洁走廊
    space = sap.Space(name, type, rectangle)
    spaces.append(space)


def get_spaces_bound(spaces):
    """
    获取spaces中最大最小的坐标点[max_xy, min_xy]
    :param spaces:
    :return:[max_xy, min_xy]
    """
    max_xy = copy.deepcopy(spaces[0].rectangle[0].point3d)
    min_xy = copy.deepcopy(spaces[0].rectangle[0].point3d)
    for index, space in enumerate(spaces):
        # if index == 0:mnm
        [temp_max, temp_min] = get_space_xy_bound(space)
        max_xy = sap.Point3D(max(temp_max.x, max_xy.x),max((temp_max.y, max_xy.y)),max(temp_max.z, max_xy.z))
        min_xy = sap.Point3D(min(min_xy.x, temp_min.x),min((min_xy.y, temp_min.y)),min(min_xy.z, temp_min.z))
        # if temp_max.x >= max_xy.x and temp_max.y >= max_xy.y:
        #     max_xy = copy.deepcopy(temp_max)
        # if temp_min.x <= min_xy.x and temp_min.y <= min_xy.y:
        #     min_xy = copy.deepcopy(temp_min)
        max_min = [max_xy, min_xy]
    return max_min


def get_space_xy_bound(space):
    """
    获取space中最大最小的坐标点,返回[max_xy, min_xy]
    :param space:
    :return:[max_xy, min_xy]
    """
    direction = space.rectangle[0].Direction
    start_point = copy.deepcopy(space.rectangle[0].point3d)
    if direction.dx != 0:
        end_point = sap.Point3D(start_point.x + float(space.rectangle[0].length) * direction.dx,
                                start_point.y + float(space.rectangle[0].width) * direction.y, 0)
    else:
        end_point = sap.Point3D(start_point.x + float(space.rectangle[0].width) * direction.x,
                                start_point.y + float(space.rectangle[0].length) * direction.dy, 0)
    min_xy = copy.deepcopy(start_point)
    max_xy = copy.deepcopy(end_point)
    # min_xy = sap.point3d(start_point.point3d.x, start_point.point3d.y, 0)
    # max_xy = sap.point3d(end_point.point3d.x, end_point.point3d.y, 0)
    min_x = min(min_xy.x, max_xy.x)
    max_x = max(min_xy.x, max_xy.x)
    min_y = min(min_xy.y, max_xy.y)
    max_y = max(min_xy.y, max_xy.y)
    min_xy = sap.Point3D(min_x, min_y, 0)
    max_xy = sap.Point3D(max_x, max_y, 0)
    # if min_xy.x > max_xy.x:
    #     temp = copy.deepcopy(min_xy)
    #     min_xy = copy.deepcopy(max_xy)
    #     max_xy = temp
    #     max_min = [max_xy, min_xy]
    return [max_xy, min_xy]

#
# values = ruffle(Aname, Atype, p, cpdPath, Anum)
# diction = get_bic_data(Aname, Anum, Aarea, Alength, Awidth, values, dict)
#
# # 一排房间,一个区域房间
# spaces = []
# # 前两个参数dxdy是说长的方向,没有存储宽的方向
# direction = sap.Direction(1.0, 0.0, 1.0, 1.0, 0.0)
# # 行开始坐标
# linePoint = sap.Point3D(0.0, 0.0, 0.0)
# # 初始化一排房间
# rowSpaces(spaces, direction, linePoint, diction.get("medicName"), diction.get("medicType"), diction.get("medicArea"),
#           diction.get("length"), diction.get("width"))
#
# paintSpaces2.draw(spaces, direction)
# # 每排修改成等宽后的代码
# aequilate(spaces, linePoint, direction)
#
# paintSpaces2.draw(spaces, direction)
#
# # 画走廊
# Clength = 12
# Cwidth = 1.5
# num = 34  # 洁净走廊
# beginPoint = copy.deepcopy(linePoint)
# addCorridor(spaces, beginPoint, direction, Clength, Cwidth, num,Aname)
#
# paintSpaces2.draw(spaces, direction)
# # 每个房间坐标移动到左上角
# Direction2Point.judge(spaces, direction)
#
# print()
