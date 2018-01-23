"""
产生初始化房间
"""
import random
import shelve

import matplotlib.pyplot as plt
import numpy as np
from datetime import time, datetime
import os
import copy

from planinitrunner.get_cpd import get_cpd, Cpd
from tool.GA_tools import isnumber

import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import planinitrunner.runtest2 as runtest2
from dao import spaceApoint as sap
from tool import csvBrief
from planinitrunner import translation, plot_draw

POP_SIZE = 1
N_GENERATIONS = 1
BOUND = np.array([[[0, 36], [-1, 11]], [[36, 81.1], [0, 81]]])  # 区域xy边界区间集合列表

S = 0  # 根据区域边界算出的总面积
BOUND_s = []  # 各个区域边界区间内的面积
for i in BOUND:
    S = S + (i[0][1]-i[0][0]) * (i[1][1] - i[1][0])
    BOUND_s.append((i[0][1] - i[0][0]) * (i[1][1] - i[1][0]))
BOUND_s = np.array(BOUND_s)
    # x = np.arange(0, 81.1, 0.1)
    # y = np.zeros((len(x), 2))
    # s = 0
    # for i in len(x):
    #     if 0 <= x[i] < 36:
    #         y[i] = [-1, 80]
    #     if 36 <= x[i] < 81.1:
    #         y[i] = [0, 81]
    #     if i != 0:
    #         s = s + ((y[i][1] - y[i][0]) + (y[i-1][1] - y[i-1][0]))*0.1*0.5


Aname = ['麻醉准备间 ', '麻醉恢复区 ', '精密仪器室 ', '无菌包储存室 ', '一次性物品储存室 ', '体外循环室 ', '麻醉用品库 ', '总护士站 ', '快速灭菌', '腔镜洗消间',
         '家属等候区 ', '患者换床间 ', '转运床停放间 ', '废弃物分类暂存间', '器械预清洗间 ', '医疗气体制备间 ', '石膏制备间 ', '病理标本间 ', '保洁用品存放间 ',
         '衣物发放及入口管理 ', '更衣 ', '换鞋间', '淋浴间', '卫生间', '二次换鞋间 ', '护士长办公室 ', '麻醉师办公室 ', '示教室 ', '休息就餐间 ', '卫生员室 ', '库房',
         '会诊室',
         '电梯', '楼梯', '洁净走廊', '清洁走廊', '管井', '前室', '前厅', '缓冲', '手术药品库 ', '保洁室 ', '谈话室', '污染被服暂存间 ', '值班室 (部分含卫生间)',
         '缓冲走廊',
         '麻醉科主任办公室 ', '一级手术室', '多功能复合手术室', '二级手术室', '三级手术室', '正负压转换手术室']
AreaAndNumber, ResultArea = runtest2.linear(S)
SPACE_NUM = AreaAndNumber
SPACE_AREA = []
dataPath2 = os.getcwd() + "/../file/ratio.csv"  # 长宽比数据
data2 = csvBrief.readListCSV(dataPath2)
data2 = data2[1:]  # 长宽数据集合
SPACE_TYPE = data2[3]
l = [float(i) if isnumber(i) else 1.0 for i in data2[0]]
LENGTH = []
w = [float(i) if isnumber(i) else 1.0 for i in data2[1]]
WIDTH = []

for i in range(len(SPACE_TYPE)):
    if SPACE_NUM[i] == 0:
        times = 1
        SPACE_AREA.append(ResultArea[i])
    else:
        times = ((float(w[i]) * float(l[i])) / (ResultArea[i] / SPACE_NUM[i])) ** 0.5
        SPACE_AREA.append(ResultArea[i] / SPACE_NUM[i])
    WIDTH.append(float(w[i]) / times)
    LENGTH.append(float(l[i]) / times)

DICT = {}
for id in range(len(SPACE_TYPE)):
    DICT[SPACE_TYPE[id]] = id

operation_index_list = [2, 3, 4, 5, 40, 47, 48, 49, 50, 51]   # 手术部各个功能房间的字典索引
operation_area = 0  # 手术部总面积
operation_num = 0
for i in operation_index_list:
    operation_num += SPACE_NUM[i]
    operation_area += SPACE_AREA[i]*SPACE_NUM[i]
operation_area_ratio_sqr = (operation_area / S) ** 0.5
bound_operation = copy.deepcopy(BOUND)
# 根据手术室占总面积情况，确定手术室的各个边界bound_operation为其边界集合
for i in range(len(BOUND)):
    x_bound = (BOUND[i, 0, 1] - BOUND[i, 0, 0]) * (1 - operation_area_ratio_sqr) / 2
    y_bound = (BOUND[i, 1, 1] - BOUND[i, 1, 0]) * (1 - operation_area_ratio_sqr) / 2
    bound_operation[i, 0] = np.array([bound_operation[i, 0, 0]+x_bound, bound_operation[i, 0, 1]-x_bound])
    bound_operation[i, 1] = np.array([bound_operation[i, 1, 0]+y_bound, bound_operation[i, 1, 1]-y_bound])


open_path = '\\..\\file\\rocky\\rocky01.txt'  # 传入cpd表对应的shelve文件的路径
cpd = Cpd(open_path)
print(cpd.state_names)
p = {}
for _ in range(operation_num):
    cpd_result = get_cpd(p, cpd)[0][1]
    # if cpd_result
# print(lll)


class GA(object):
    def __init__(self, bound, pop_size, name_list, width, length, index_dict, type_list, space_num_list,
                 spaces_type_list, bound_s, s, operation_index_list, cpd):
        # self.DNA_size = DNA_size
        # self.cross_rate = cross_rate
        self.bound = bound
        self.bound_s = bound_s
        self.s = s
        self.operation_index_list = operation_index_list
        self.pop_size = pop_size
        self.name_list = name_list
        self.width = width
        self.length = length
        self.index_dict = index_dict
        self.type_list = type_list
        self.space_num_list = space_num_list
        self.spaces_type_list = spaces_type_list
        all_species = []

        # for _ in range(pop_size):
        #     all_spaces = []
        #     dict = {}
        #     for index, space_num in enumerate(space_num_list):
        operation_num_list = [space_num_list[i] for i in operation_index_list]

        for _ in range(pop_size):
            all_spaces = []
            dict = {}
            # for id in range(len(name_list)):
            #     dict[Atype[id]] = id
            for index in operation_index_list:
                space_num = space_num_list[index]
                if space_num != 0:
                    # 根据区间概率，随机选取产生space_num所在的区域区间
                    idx = np.random.choice(len(bound_s), size=space_num, replace=True, p=bound_s / s)
                    for bd in idx:
                        bd = bound[bd]
                        position1 = np.random.rand(space_num, 1) * (bd[0][1]-bd[0][0]) + bd[0][0]
                        position2 = np.random.rand(space_num, 1) * (bd[1][1]-bd[1][0]) + bd[1][0]
                        position = np.concatenate((position1, position2), axis=1)
                    for i in range(space_num):
                        begin_point = sap.Point3D(position[i][0], position[i][1], 0)
                        # 根据给定的spaces_type_list对应的type确定其space_type
                        space_type = spaces_type_list[index]
                        index_csv = index_dict[space_type]
                        name = name_list[index_csv]+str(len(all_spaces))
                        # area = 30
                        # times = (length[index_csv]*width[index_csv]/area)**0.5
                        # space_len = length[index_csv]/times
                        # space_width = width[index_csv]/times
                        direction = sap.Direction(1, 0, 1, 1, 0)
                        runtest2.addOneSpace(all_spaces, begin_point, width[index], length[index], name, space_type, direction)
            all_species.append(all_spaces)

        # for _ in range(pop_size):
        #     all_spaces = []
        #     dict = {}
        #     # for id in range(len(name_list)):
        #     #     dict[Atype[id]] = id
        #     for index in operation_index_list:
        #         space_num = space_num_list[index]
        #         if space_num != 0:
        #             # 根据区间概率，随机选取产生space_num所在的区域区间
        #             idx = np.random.choice(len(bound_s), size=space_num, replace=True, p=bound_s / s)
        #             for bd in idx:
        #                 bd = bound[bd]
        #                 position1 = np.random.rand(space_num, 1) * (bd[0][1]-bd[0][0]) + bd[0][0]
        #                 position2 = np.random.rand(space_num, 1) * (bd[1][1]-bd[1][0]) + bd[1][0]
        #                 position = np.concatenate((position1, position2), axis=1)
        #             for i in range(space_num):
        #                 begin_point = sap.Point3D(position[i][0], position[i][1], 0)
        #                 # 根据给定的spaces_type_list对应的type确定其space_type
        #                 space_type = spaces_type_list[index]
        #                 index_csv = index_dict[space_type]
        #                 name = name_list[index_csv]+str(len(all_spaces))
        #                 # area = 30
        #                 # times = (length[index_csv]*width[index_csv]/area)**0.5
        #                 # space_len = length[index_csv]/times
        #                 # space_width = width[index_csv]/times
        #                 direction = sap.Direction(1, 0, 1, 1, 0)
        #                 runtest2.addOneSpace(all_spaces, begin_point, width[index], length[index], name, space_type, direction)
        #     all_species.append(all_spaces)
        self.all_species = all_species

    # def get_cost(self, all_species, bound, length, width):
    #     all_species_cost = []
    #     for all_spaces in all_species:
    #         space_max_min = []
    #         cross_cost = 0
    #         cross_cost_index = set()
    #         size_cost = 0
    #         size_cost_index = set()
    #         over_bound = 0
    #         over_bound_index = set()
    #         # 重叠的损失计算/超越边界计算
    #         for index, space in enumerate(all_spaces):
    #             [max_xy, min_xy] = runtest2.get_space_xy_bound(space)
    #             sec_point = sap.Point3D(max_xy.x, min_xy.y, max_xy.z)
    #             four_point = sap.Point3D(min_xy.x, max_xy.y, max_xy.z)
    #             d_x = max_xy.x - min_xy.x
    #             d_y = max_xy.y - min_xy.y
    #             space_max_min.append([min_xy, sec_point, max_xy, four_point, d_x, d_y])
    #             if bound:
    #                 # 超越边界计算
    #                 max_x = max(bound[0][0][1], max_xy.x)
    #                 max_y = max(bound[0][1][1], max_xy.y)
    #                 min_x = min(bound[0][0][0], min_xy.x)
    #                 min_y = min(bound[0][1][0], min_xy.y)
    #                 cross_x = (max_x - min_x) - (bound[0][0][1] - bound[0][0][0]) - d_x
    #                 cross_y = (max_y - min_y) - (bound[0][1][1] - bound[0][1][0]) - d_y
    #                 if cross_x > -d_x or cross_y > -d_y:
    #                     over_bound = over_bound + abs(cross_x) + abs(cross_y)
    #                     over_bound_index.add(index)
    #         # 重叠的损失计算
    #         for index1, space1 in enumerate(all_spaces):
    #             for index2, space2 in enumerate(all_spaces):
    #                 if index1 < index2:
    #                     max_min = runtest2.get_spaces_bound([space1, space2])
    #                     cross_x = (max_min[0].x - max_min[1].x) - space_max_min[index1][4] - space_max_min[index2][4]
    #                     cross_y = (max_min[0].y - max_min[1].y) - space_max_min[index1][5] - space_max_min[index2][5]
    #                     if cross_x < 0 and cross_y < 0:
    #                         cross_cost = cross_cost + abs(cross_x) + abs(cross_y)
    #                         cross_cost_index.add(index1)
    #         # # 尺寸损失计算
    #         # dataPath2 = os.getcwd() + "/../file/ratio.csv"  # 长宽比数据
    #         # data2 = csvBrief.readListCSV(dataPath2)
    #         # data2 = data2[1:]  # 长宽数据集合
    #         #
    #         # dict = {}
    #         # for id in range(len(data2[0])):
    #         #     dict[data2[3][id]] = id
    #         # for index, space in enumerate(all_spaces):
    #         #     index_csv = dict[space.type]
    #         #     a = length[index_csv]
    #         #     b = width[index_csv]
    #         #     c = length[index_csv]
    #         #     d = width[index_csv]
    #         #     lw_ratio_csv = max(float(length[index_csv]), float(width[index_csv])) / min(float(length[index_csv]),
    #         #                                                                                 float(width[index_csv]))
    #         #     lw_ratio = max(d_x, d_y) / min(d_x, d_y)
    #         #     if lw_ratio < lw_ratio_csv - 0.3 or lw_ratio > lw_ratio_csv + 0.3:
    #         #         size_cost = size_cost + abs(lw_ratio - lw_ratio_csv)
    #         #         size_cost_index.add(index)
    #         sum_cost = (cross_cost + over_bound)
    #         individual = [sum_cost, cross_cost, cross_cost_index, over_bound, over_bound_index]
    #         all_species_cost.append(individual)
    #     return all_species_cost
    #
    # def mutate(self, bound, all_species_cost, all_species):
    #     min_index = np.argmin(np.array(all_species_cost)[:, 0])
    #     # max_index = np.argmax(np.array(all_species_cost)[:, 0])
    #     all_species.append(copy.deepcopy(all_species[min_index]))
    #     # all_species_cost.append(all_species_cost[min_index])
    #     for index, individual in enumerate(all_species_cost):
    #         all_spaces = all_species[index]
    #         [sum_cost, cross_cost, cross_cost_index, over_bound, over_bound_index] = individual
    #         # if sum_cost > all_species_cost[min_index][0] and random.randint(1, 9) % 3 == 0:
    #         if sum_cost > all_species_cost[min_index][0] or index == len(all_species_cost):
    #
    #             for space_index in cross_cost_index:
    #                 space = all_spaces[space_index]
    #                 spaces = [space]
    #                 [max_xy, min_xy] = runtest2.get_space_xy_bound(space)
    #                 translation.translation(spaces,
    #                                         random.uniform(bound[0][0][0] - min_xy.x,
    #                                                        bound[0][0][1] - max_xy.x),
    #                                         random.uniform(bound[0][1][0] - min_xy.y,
    #                                                        bound[0][1][1] - max_xy.y))
    #             for space_index in over_bound_index:
    #                 space = all_spaces[space_index]
    #                 spaces = [space]
    #                 [max_xy, min_xy] =runtest2.get_space_xy_bound(space)
    #                 translation.translation(spaces,
    #                                         random.uniform(bound[0][0][0] - min_xy.x,
    #                                                        bound[0][0][1] - max_xy.x),
    #                                         random.uniform(bound[0][1][0] - min_xy.y,
    #                                                        bound[0][1][1] - max_xy.y))
    #         all_species[index] = all_spaces
    #
    #         # for point in range(self.DNA_size):
    #         #     if np.random.rand() < self.mutate_rate:
    #         #         # 随机选取两个点对调位置
    #         #         swap_point = np.random.randint(0, self.DNA_size)
    #         #         swapA, swapB = child[point], child[swap_point]
    #         #         child[point], child[swap_point] = swapB, swapA
    #     return all_species
        # def translateDNA(self, DNA, city_position):     # get cities' coord in order
        #     line_x = np.empty_like(DNA, dtype=np.float64)
        #     line_y = np.empty_like(DNA, dtype=np.float64)
        #     for i, d in enumerate(DNA):
        #         city_coord = city_position[d]
        #         line_x[i, :] = city_coord[:, 0]
        #         line_y[i, :] = city_coord[:, 1]
        #     return line_x, line_y

        # def get_fitness(self, line_x, line_y):
        # total_distance = np.empty((line_x.shape[0],), dtype=np.float64)
        # for i, (xs, ys) in enumerate(zip(line_x, line_y)):
        #     # diff函数就是执行的是后一个元素减去前一个元素
        #     total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
        # fitness = np.exp(self.DNA_size * 2 / total_distance)
        # return fitness, total_distance

        #     def select(self, fitness):
        #         idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        #         return self.pop[idx]
        #
        #     def crossover(self, parent, pop):
        #         if np.random.rand() < self.cross_rate:
        #             i_ = np.random.randint(0, self.pop_size, size=1)  # select another individual from pop
        #             cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)  # choose crossover points
        #             keep_city = parent[~cross_points]  # find the city number
        #             # swap_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert=True)]
        #             swap_city = pop[i_, np.in1d(pop[i_].ravel(), keep_city, invert=True)]
        #
        #             parent[:] = np.concatenate((keep_city, swap_city))
        #         return parent
        #

    #
    # def evolve(self, all_species, bound):
    #     # pop_copy = pop.copy()
    #     # for parent in pop:  # for every parent
    #     # child = self.crossover(parent, pop_copy)
    #     all_species_cost = self.get_cost(all_species, bound, length, width)
    #     all_species = self.mutate(bound, all_species_cost, all_species)
    #     self.all_species = all_species


class TravelSalesPerson(object):
    def __init__(self, best_idx, all_species):
        self.best_idx = best_idx
        self.all_species = all_species

        # plt.ion()

    def plotting(self, best_idx, all_species):
        plt.cla()
        spaces = all_species[best_idx]
        # direction = sap.Direction(1, 0, 1, 1, 0)
        plot_draw.GA_draw_data(spaces)
        plt.pause(0.0000000000000000000001)


ga = GA(BOUND, POP_SIZE, Aname, WIDTH, LENGTH, DICT, SPACE_TYPE, SPACE_NUM, SPACE_TYPE, BOUND_s, S,
        operation_index_list, cpd)
starttime = datetime.now()
print(starttime)
for generation in range(N_GENERATIONS):
    all_species = ga.all_species
    bound = ga.bound
    # lx, ly分别为每一代中x,y坐标矩阵，每一行为每个个体对应点坐标
    # lx, ly = ga.translateDNA(ga.pop, env.city_position)

    # # 通过计算每个个体中点的距离和，将距离和作为惩罚系数（除数）获取适应度，返回适应度及总距离
    # all_species_cost = ga.get_cost(all_species, bound, ga.length, ga.width)
    # # 进化过程主要
    # ga.mutate(bound, all_species_cost, all_species)
    # all_species_cost = np.array(all_species_cost)
    # best_idx = np.argmin(all_species_cost[:, 0])

    # print('Gen:', generation, 'best individual is:', best_idx, '| best fit: %.2f' % all_species_cost[best_idx][0], )
    # env = TravelSalesPerson(best_idx, all_species)
    env = TravelSalesPerson(0, all_species)
    env.plotting(0, all_species)
    # if all_species_cost[best_idx][0] == 0:
    #     break
    #endtime = datetime.now()
    #print(endtime)
    #print((endtime - starttime).seconds)
    plt.pause(0.00001)
endtime = datetime.now()
print(endtime)
print((endtime - starttime).seconds)

plt.ioff()
plt.show()
