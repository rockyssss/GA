"""
Visualize Genetic Algorithm to find the shortest path for travel sales problem.

Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/
"""
import random
from dao.rectangle_index import *
import matplotlib.pyplot as plt
import numpy as np
# from datetime import time, datetime
from timeit import default_timer as time
import os
import copy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
from tool.GA_tools import isnumber, new_rectangle, new_ros

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from numba import jit
import planinitrunner.runtest2 as runtest2
from dao import GAspaceApoint2 as sap
from tool import csvBrief, translation
from planinitrunner import translation, plot_draw


def distance_2_m(np_arr, accuracy):
    np_arr = np.around(np_arr / accuracy, decimals=0).astype(int)
    return np_arr


DT = 1
POP_SIZE = 15
ACCUR = 0.3
N_GENERATIONS = 100000
SPACE_TYPE = np.array(['51449600', '51446500', '51442113', '51171300', '51543115', '51172100','51446300', '51348400',
                       '51446100', '51448700']).astype(int)
# SPACE_NUM = [2, 2, 2, 2, 2, 2, 2, 2, 2, 0]
SPACE_NUM = np.array([2, 2, 0, 0, 0, 0, 0, 0, 0, 0])
COLUMN_NUM = 12
DG = {51449600: 3, 51446500: 2, 51442113: 1}
D_EN = np.array([5, 0, 0])
D_EN_M = distance_2_m(D_EN, ACCUR)
P_EN = np.array([5, 25, 0])
P_EN_M = distance_2_m(P_EN, ACCUR)
# BOUND = np.array([[[0, 50], [0, 25]]])
BOUND = np.array([[[0, 36], [0, 20]], [[36, 50], [0, 25]]])
BOUND_M = distance_2_m(BOUND, ACCUR)

S = 0  # 根据区域边界算出的总面积
BOUND_s = np.array([], dtype=int)  # 各个区域边界区间内的面积
for i in BOUND:
    si = (i[0][1] - i[0][0]) * (i[1][1] - i[1][0])
    si = distance_2_m(si, ACCUR)
    S = S + si
    # BOUND_s = np.append(BOUND_s, np.array([si]))
    BOUND_s = np.concatenate((BOUND_s, np.array([si])), axis=0)

AreaAndNumber, ResultArea = runtest2.linear(S)
ResultArea = [i if i > 0 else 0 for i in ResultArea]
ResultArea = distance_2_m(np.array(ResultArea), ACCUR)

ROOMS_NUM = 40
Aname = np.array(['麻醉准备间 ', '麻醉恢复区 ', '精密仪器室 ', '无菌包储存室 ', '一次性物品储存室 ', '体外循环室 ', '麻醉用品库 ', '总护士站 ', '快速灭菌', '腔镜洗消间',
         '家属等候区 ', '患者换床间 ', '转运床停放间 ', '废弃物分类暂存间', '器械预清洗间 ', '医疗气体制备间 ', '石膏制备间 ', '病理标本间 ', '保洁用品存放间 ',
         '衣物发放及入口管理 ', '更衣 ', '换鞋间', '淋浴间', '卫生间', '二次换鞋间 ', '护士长办公室 ', '麻醉师办公室 ', '示教室 ', '休息就餐间 ', '卫生员室 ', '库房',
         '会诊室',
         '电梯', '楼梯', '洁净走廊', '清洁走廊', '管井', '前室', '前厅', '缓冲', '手术药品库 ', '保洁室 ', '谈话室', '污染被服暂存间 ', '值班室 (部分含卫生间)',
         '缓冲走廊',
         '麻醉科主任办公室 ', '一级手术室', '多功能复合手术室', '二级手术室', '三级手术室', '正负压转换手术室'])

dataPath2 = os.getcwd() + "/../file/ratio.csv"  # 长宽比数据
data2 = csvBrief.readListCSV(dataPath2)
data2 = data2[1:]  # 长宽数据集合
l = [float(i) if isnumber(i) else 1.0 for i in data2[0]]
LENGTH = []
w = [float(i) if isnumber(i) else 1.0 for i in data2[1]]
WIDTH = []
SPACE_AREA = []
for i in range(len(SPACE_TYPE)):
    if SPACE_NUM[i] == 0 or ResultArea[i] == 0:
        times = 1
        SPACE_AREA.append(0)
    else:
        times = ((w[i] * l[i]) / (ResultArea[i] / SPACE_NUM[i])) ** 0.5
        SPACE_AREA.append(ResultArea[i] / SPACE_NUM[i])
    WIDTH.append(float(w[i]) / times)
    LENGTH.append(float(l[i]) / times)
SPACE_AREA = np.array(SPACE_AREA)
WIDTH = np.array(WIDTH)
LENGTH = np.array(LENGTH)
WIDTH_M = distance_2_m(WIDTH, ACCUR)
LENGTH_M = distance_2_m(LENGTH, ACCUR)

TYPE = np.array(data2[3])
TYPE = TYPE.astype(int)
DICT = {}
for id in range(len(TYPE)):
    DICT[TYPE[id]] = id


class GA(object):
    def __init__(self, bound, pop_size, name_list, width, length, index_dict, type_list, space_num_list,
                 spaces_type_list, d_en, p_en, dg, bound_s, s, column_num):
        # self.DNA_size = DNA_size
        # self.cross_rate = cross_rate
        self.bound = bound
        self.bound_s = bound_s
        self.s = s
        self.pop_size = pop_size
        self.name_list = name_list
        self.width = width
        self.length = length
        self.index_dict = index_dict
        self.type_list = type_list
        self.space_num_list = space_num_list
        self.spaces_type_list = spaces_type_list
        self.d_en = d_en
        self.p_en = p_en
        self.dg = dg
        self.column_num = column_num
        # self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)])
        # space_point = np.vstack([[np.random.rand(rooms_num, 2)*500] for _ in range(pop_size)])
        sum_space_num = space_num_list.sum()
        all_spaces = np.zeros([sum_space_num, column_num], dtype=int)
        all_species = np.zeros([pop_size, sum_space_num, column_num], dtype=int)
        for pop_index in range(pop_size):

            dict = {}
            # for id in range(len(name_list)):
            #     dict[Atype[id]] = id
            space_id = 0

            for index, space_num in enumerate(space_num_list):
                if space_num != 0:
                    # 根据区间概率，随机选取产生space_num所在的区域区间
                    idx = np.random.choice(len(bound_s), size=1, replace=True, p=bound_s / s)
                    space_type = spaces_type_list[index]
                    w = width[index]
                    l = length[index]
                    same_space = np.zeros([space_num, 2], dtype=int)
                    for index,bd in enumerate(space_num):
                        bd = bound[idx]
                        point = [np.random.randint(bd[0][0], bd[0][1]-l), np.random.randint(bd[1][0], bd[1][1]-w)]
                        same_space[index] = np.array(point)
                    for i in range(space_num):
                        # index_csv = index_dict[space_type]
                        x = same_space[i][X_MIN_INDEX]
                        y = same_space[i][Y_MAX_INDEX]
                        xm = x + l
                        ym = y + w
                        space = new_ros(x, y, xm, ym, l, w, space_id, space_type, -25, -25, -25, -25, idx)

                        # 根据给定的spaces_type_list对应的type确定其space_type
                        # direction = sap.Direction(1, 0, 1, 1, 0)
                        all_spaces[space_id] = space
                        space_id += 1

            all_species[pop_index] = all_spaces
        self.all_species = all_species

    # def route_cost(self,all_species,doctor,patien):

    # @jit
    def get_cost(self, all_species, bound, length, width, p_en, d_en, dg):
        # bound = [[[0, 500], [0, 500]]]
        d = []
        all_species_cost = []
        for all_spaces in all_species:
            space_max_min = []
            cross_cost = 0
            cross_cost_index = set()
            size_cost = 0
            size_cost_index = set()
            over_bound = 0
            over_bound_index = set()
            route_cost = 0
            route_cost_dict = dict()
            # 重叠的损失计算

            # 重叠的损失计算/超越边界计算
            for index1, space1 in enumerate(all_spaces):

                # sec_point = sap.Point3D(max_xy.x, min_xy.y, max_xy.z)
                # four_point = sap.Point3D(min_xy.x, max_xy.y, max_xy.z)
                # d_x = max_xy.x - min_xy.x
                # d_y = max_xy.y - min_xy.y
                # space_max_min.append([min_xy, sec_point, max_xy, four_point, d_x, d_y])
                if len(bound):
                    max_x1 = np.argmin(space1[:, X_MAX_INDEX])
                    max_y1 = np.argmin(space1[:, Y_MAX_INDEX])
                    min_x1 = np.argmin(space1[:, X_MIN_INDEX])
                    min_y1 = np.argmin(space1[:, Y_MIN_INDEX])
                    bound_index = space1[0][BOUND_INDEX]
                    # 超越边界计算
                    max_xs = max(bound[bound_index][0][1], max_x1)
                    max_ys = max(bound[bound_index][1][1], max_y1)
                    min_xs = min(bound[bound_index][0][0], min_x1)
                    min_ys = min(bound[bound_index][1][0], min_y1)
                    cross_x = (max_xs - min_xs) - (bound[bound_index][0][1] - bound[bound_index][0][0])
                    cross_y = (max_ys - min_ys) - (bound[bound_index][1][1] - bound[bound_index][1][0])
                    if cross_x > 0 or cross_y > 0:
                        over_bound = over_bound + cross_x + cross_y
                        over_bound_index.add(index1)

                # 动线计算
                cost = abs(min_x1 - d_en[0]) + abs(min_x1 - p_en[0]) + abs(min_y1 - d_en[1]) + abs(min_y1 - p_en[1])
                if cross_cost < 10:
                    cost = cost*dg[space1.type]
                else:
                    cost = cost * dg[space1.type]*0.4
                if space1.type in route_cost_dict.keys():
                    if cost > route_cost_dict[space1[TYPE_INDEX]][0]:
                        route_cost_dict[space1[TYPE_INDEX]] = [cost, index1]
                else:
                    route_cost_dict[space1[TYPE_INDEX]] = [cost, index1]
                route_cost += cost

            # 重叠计算
            # for index1, space1 in enumerate(all_spaces):
                for index2, space2 in enumerate(all_spaces):
                    if index1 < index2:
                        max_x2 = np.argmin(space2[:, X_MAX_INDEX])
                        max_y2 = np.argmin(space2[:, Y_MAX_INDEX])
                        min_x2 = np.argmin(space2[:, X_MIN_INDEX])
                        min_y2 = np.argmin(space2[:, Y_MIN_INDEX])
                        max_xc = np.max([max_x1, max_x2])
                        max_yc = np.max([max_y1, max_y2])
                        min_xc = np.min([min_x1, min_x2])
                        min_yc = np.max([min_y1, min_y2])
                        d_x1 = max_x1 - min_x1
                        d_x2 = max_x2 - min_x2
                        d_y1 = max_y1 - min_y1
                        d_y2 = max_y2 - min_y2
                        # max_min = runtest2.get_spaces_bound([space1, space2])
                        cross_x = (max_xc - min_xc) - d_x1 - d_x2
                        cross_y = (max_yc - min_yc) - d_y1 - d_y2
                        if cross_x < 0 and cross_y < 0:
                            cross_cost = cross_cost - cross_x - cross_y
                            cross_cost_index.add(index1)


            # # 尺寸损失计算
            # dataPath2 = os.getcwd() + "/../file/ratio.csv"  # 长宽比数据
            # data2 = csvBrief.readListCSV(dataPath2)
            # data2 = data2[1:]  # 长宽数据集合
            #
            # dict = {}
            # for id in range(len(data2[0])):
            #     dict[data2[3][id]] = id
            # for index, space in enumerate(all_spaces):
            #     index_csv = dict[space.type]
            #     a = length[index_csv]
            #     b = width[index_csv]
            #     c = length[index_csv]
            #     d = width[index_csv]
            #     lw_ratio_csv = max(float(length[index_csv]), float(width[index_csv])) / min(float(length[index_csv]),
            #                                                                                 float(width[index_csv]))
            #     lw_ratio = max(d_x, d_y) / min(d_x, d_y)
            #     if lw_ratio < lw_ratio_csv - 0.3 or lw_ratio > lw_ratio_csv + 0.3:
            #         size_cost = size_cost + abs(lw_ratio - lw_ratio_csv)
            #         size_cost_index.add(index)
            sum_cost = (cross_cost + over_bound + route_cost)
            individual = [sum_cost, cross_cost, cross_cost_index, over_bound, over_bound_index, route_cost_dict]
            all_species_cost.append(individual)
        return all_species_cost

    # @jit
    def mutate(self, bound, all_species_cost, all_species, dt):
        min_index = np.argmin(np.array(all_species_cost)[:, 0])
        max_index = np.argmax(np.array(all_species_cost)[:, 0])
        # all_species.append(copy.deepcopy(all_species[min_index]))
        # all_species_cost.append(all_species_cost[min_index])
        for index, individual in enumerate(all_species_cost):
            all_spaces = all_species[index]
            [sum_cost, cross_cost, cross_cost_index, over_bound, over_bound_index, route_cost_dict] = individual
            if sum_cost > all_species_cost[min_index][0]:
            # if sum_cost > all_species_cost[min_index][0] and random.randint(1, 9) % 3 == 0:
                # if sum_cost > all_species_cost[min_index][0] or index == len(all_species_cost):
                st = list(route_cost_dict.values())
                route_cost = set({int(i) for i in np.array(st)[:, 1]})
                all_move_set = route_cost | cross_cost_index | over_bound_index
                for space_index in all_move_set:
                        space = all_spaces[space_index]
                        spaces = [space]
                        [max_xy, min_xy] = runtest2.get_space_xy_bound(space)
                        translation(spaces,
                                                random.uniform(bound[0][0][0] - min_xy.x,
                                                               bound[0][0][1] - max_xy.x) * dt,
                                                random.uniform(bound[0][1][0] - min_xy.y,
                                                               bound[0][1][1] - max_xy.y) * dt)
                # for space_index in over_bound_index:
                #     space = all_spaces[space_index]
                #     spaces = [space]
                #     [max_xy, min_xy] = runtest2.get_space_xy_bound(space)
                #     translation.translation(spaces,
                #                             random.uniform(bound[0][0][0] - min_xy.x,
                #                                            bound[0][0][1] - max_xy.x) * dt,
                #                             random.uniform(bound[0][1][0] - min_xy.y,
                #                                            bound[0][1][1] - max_xy.y) * dt)
                # for i in route_cost_dict.values():
                #     space = all_spaces[i[1]]
                #     spaces = [space]
                #     [max_xy, min_xy] = runtest2.get_space_xy_bound(space)
                #     translation.translation(spaces,
                #                             random.uniform(bound[0][0][0] - min_xy.x,
                #                                            bound[0][0][1] - max_xy.x) * dt,
                #                             random.uniform(bound[0][1][0] - min_xy.y,
                #                                            bound[0][1][1] - max_xy.y) * dt)

            all_species[index] = all_spaces

            # for point in range(self.DNA_size):
            #     if np.random.rand() < self.mutate_rate:
            #         # 随机选取两个点对调位置
            #         swap_point = np.random.randint(0, self.DNA_size)
            #         swapA, swapB = child[point], child[swap_point]
            #         child[point], child[swap_point] = swapB, swapA
        return all_species


    def evolve(self, all_species, bound):
        # pop_copy = pop.copy()
        # for parent in pop:  # for every parent
        # child = self.crossover(parent, pop_copy)
        all_species_cost = self.get_cost(all_species, bound, length, width)
        all_species = self.mutate(bound, all_species_cost, all_species)
        self.all_species = all_species


class TravelSalesPerson(object):
    def __init__(self, best_idx, all_species):
        self.best_idx = best_idx
        self.all_species = all_species

        # plt.ion()

    # @jit
    def plotting(self, best_idx, all_species):
        # plt.cla()
        spaces = all_species[best_idx]
        # direction = sap.Direction(1, 0, 1, 1, 0)
        plot_draw.GA_draw_data(spaces)
        plt.pause(0.0000000000000000000001)


ga = GA(BOUND_M, POP_SIZE, Aname, WIDTH_M, LENGTH_M, DICT, TYPE, SPACE_NUM, SPACE_TYPE, D_EN_M, P_EN_M, DG, BOUND_s, S,
        COLUMN_NUM)
starttime = time()
print(starttime)
t = 1
for generation in range(N_GENERATIONS):
    gs = time()
    all_species = ga.all_species
    bound = ga.bound
    t = t * DT
    # lx, ly分别为每一代中x,y坐标矩阵，每一行为每个个体对应点坐标
    # lx, ly = ga.translateDNA(ga.pop, env.city_position)
    # 通过计算每个个体中点的距离和，将距离和作为惩罚系数（除数）获取适应度，返回适应度及总距离
    all_species_cost = ga.get_cost(all_species, bound, ga.length, ga.width, ga.p_en, ga.d_en, ga.dg)
    # 进化过程主要
    ga.mutate(bound, all_species_cost, all_species, t)
    all_species_cost = np.array(all_species_cost)
    best_idx = np.argmin(all_species_cost[:, 0])
    print('Gen:', generation, 'best individual is:', best_idx, '| best fit: %.2f' % all_species_cost[best_idx][0], )
    env = TravelSalesPerson(best_idx, all_species)
    env.plotting(best_idx, all_species)
    if all_species_cost[best_idx][0] == 250:
        break

    gd = time()
    plt.pause(0.00001)
    print(gd - gs)
endtime = time()
print(endtime)
print(endtime - starttime)

plt.ioff()
plt.show()
