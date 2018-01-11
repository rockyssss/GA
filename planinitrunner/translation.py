#!/usr/bin/python
# _*_ coding: UTF-8 _*_
import os
import shelve
import math
from dao import spaceApoint as sap
# from planinitrunner import paintSpaces2, get_cpd, tubeWell
from tool import csvBrief, other, widthRectangles, spacesChanges, corridor, Direction2Point
from tool import lengthAndWidth as lw
import planinitrunner.runtest2 as  run2
import copy
import random
# import planinit.plot_spaces
# from planinit.ptah_config import *
"""
    画电梯
"""
spaces = []
allspaces = []


def elevator(direction, linePoint, y):
    areaAndNumber, ResultArea = run2.linear(y)
    # 测试全流程
    dataPath2 = RATIO_PATH  # 长宽比数据
    dict2 = csvBrief.readListCSV(dataPath2)
    data2 = dict2[1:]  # 长宽数据集合
    # Aname = dict2[0]  # 数据的列名称集合
    Anum = areaAndNumber
    Aname = ['麻醉准备间 ', '麻醉恢复区 ', '精密仪器室 ', '无菌包储存室 ', '一次性物品储存室 ', '体外循环室 ', '麻醉用品库 ', '总护士站 ', '快速灭菌', '腔镜洗消间',
             '家属等候区 ',
             '患者换床间 ', '转运床停放间 ', '废弃物分类暂存间', '器械预清洗间 ', '医疗气体制备间 ', '石膏制备间 ', '病理标本间 ', '保洁用品存放间 ', '衣物发放及入口管理 ',
             '更衣 ', '换鞋间',
             '淋浴间', '卫生间', '二次换鞋间 ', '护士长办公室 ', '麻醉师办公室 ', '示教室 ', '休息就餐间 ', '卫生员室 ', '库房', '会诊室', '电梯', '楼梯', '洁净走廊',
             '清洁走廊',
             '管井', '前室', '前厅', '缓冲', '手术药品库 ', '保洁室 ', '谈话室', '污染被服暂存间 ', '值班室 (部分含卫生间)', '缓冲走廊', '麻醉科主任办公室 ', '一级手术室',
             '多功能复合手术室', '二级手术室', '三级手术室', '正负压转换手术室']
    Aarea = ResultArea
    print(Anum)
    print(Aarea)
    Alength = data2[0]
    Awidth = data2[1]
    Atype = data2[3]
    dict = {}
    for id in range(len(Aname)):
        dict[Atype[id]] = id
    # allspaces存储所有房间
    rlength = 0
    beginPoint = sap.Point3D(0, 0, 0)
    Clength = 12
    rlength = Clength
    Cwidth = 10
    name = Aname[33]
    type = "23111300"

    run2.addOneSpace(spaces, beginPoint, Clength, Cwidth, name, type, direction)
    allspaces.extend(spaces)
    return allspaces
    # paintSpaces2.draw(allspaces, direction)


# direction = sap.Direction(1.0, 0.0, 1.0, -1.0, 0.0)
# linePoint = sap.Point3D(52.594067796610176, 22.081944444444442, 0.0)
# spaces4 = elevator(direction, linePoint, 3000)


# 区的平移
def translation(spaces, x, y):
    for space in spaces:
        space.rectangle[0].point3d.x = space.rectangle[0].point3d.x + x
        space.rectangle[0].point3d.y = space.rectangle[0].point3d.y + y
    return spaces


# translation(spaces, 10, 2)
#
# paintSpaces2.draw(allspaces, direction)


# 缩放
def zoom(spaces, ratio):
    for space in spaces:
        space.rectangle[0].length = space.rectangle[0].length * ratio
        space.rectangle[0].width = space.rectangle[0].width * ratio
    return spaces


# zoom(spaces, 0.1)
#
# paintSpaces2.draw(allspaces, direction)


# 旋转
def spin(spaces, angle, quadrant):
    for space in spaces:
        if (quadrant == 1):
            space.rectangle[0].Direction.x = 1 * math.asin(angle)
            space.rectangle[0].Direction.y = 1 * math.acos(angle)
        elif (quadrant == 2):
            space.rectangle[0].Direction.x = -1 * math.asin(angle)
            space.rectangle[0].Direction.y = -1 * math.acos(angle)
        elif (quadrant == 3):
            space.rectangle[0].Direction.x = -1 * math.asin(angle)
            space.rectangle[0].Direction.y = -1 * math.acos(angle)
        else:
            space.rectangle[0].Direction.x = -1 * math.asin(angle)
            space.rectangle[0].Direction.y = 1 * math.acos(angle)
    return spaces


# spin(spaces, 30)


# 镜像
def image(spaces, coordinate):
    for space in spaces:
        if (coordinate == 'x'):
            space.rectangle[0].point3d.y = -space.rectangle[0].point3d.y
        else:
            space.rectangle[0].point3d.x = -space.rectangle[0].point3d.x
    return spaces


# image(spaces, 'x')
# paintSpaces2.draw(allspaces, direction)
