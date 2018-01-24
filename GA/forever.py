# -*- coding: utf-8 -*-
# !/bin/env python

from timeit import Timer
import numpy
import random
from numba import jit
import numpy
from timeit import Timer

obj_list = []
a = numpy.arange(27).reshape(9, 3)

class Point3D:
    def __init__(self, x, y, z):
        """
        某个图形的三维坐标参考起始点
        :param x: 参考起始点x坐标
        :param y: 参考起始点y坐标
        :param z: 参考起始点z坐标
        """
        self.x = x
        self.y = y
        self.z = z


def init_objs(num):
    normal_ary = numpy.random.random((num, 3))
    for i in range(num):
        p = Point3D(normal_ary[i, 0], normal_ary[i, 1], normal_ary[i, 2])
        obj_list.append(p)
    return normal_ary


def test1():
    s1 = sum_obj_array(obj_array)
    return s1


def test2():
    s2 = sum_array(random_array)
    return s2


def test3():
    s1 = jit_sum_obj_array(obj_array)
    return s1


def test4():
    s2 = jit_sum_array(random_array)
    return s2


def test5():
    sum = sum2d(random_array)
    return sum


def test6():
    sum = sum2d_normal(random_array)
    return sum


def sum_obj_array(array):
    sum = 0.0
    for p in array:
        sum += p.x + p.y + p.z
        # sum += p.x
    return sum


def sum_array(array):
    sum = 0.0
    for p in array:
        # print(p)
        sum += p[0] + p[1] + p[2]
    return sum


@jit
def jit_sum_obj_array(array):
    sum = 0.0
    for p in array:
        sum += p.x + p.y + p.z
    return sum


@jit
def jit_sum_array(array):
    sum = 0.0
    M, N = array.shape
    for p in range(M):
        # print(p)
        sum += array[p][1] + array[p][2] + array[p][0]
        # sum += p[0]
    return sum


@jit
def sum2d(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result


# @jit
def sum2d_normal(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result


if __name__ == '__main__':
    array_size = 10000
    intr_num = 1000
    random_array = init_objs(array_size)
    obj_array = numpy.array(obj_list)
    new_list = []
    for c in random_array:
        new_list.append(list(c))

    # print(test1())
    # print(test2())
    # print(test3())
    # print(test4())
    # print(test5())
    t1 = Timer("test1()", "from __main__ import test1")
    t2 = Timer("test2()", "from __main__ import test2")
    t3 = Timer("test3()", "from __main__ import test3")
    t4 = Timer("test4()", "from __main__ import test4")
    t5 = Timer("test5()", "from __main__ import test5")
    t6 = Timer("test6()", "from __main__ import test6")

    print("sum obj ", t1.timeit(intr_num))
    print("sum ary ", t2.timeit(intr_num))
    print("jit sum obj ", t3.timeit(intr_num))
    print("jit sum ary ", t4.timeit(intr_num))
    print("jit sum ", t5.timeit(intr_num))
    print("normal sum ", t6.timeit(intr_num))
    # print(t3.timeit(1000000))
    # print(t1.repeat(3, 1000000))
    # print(t2.repeat(3, 1000000))
    # print(t3.repeat(3, 1000000))


    print()



        # self.uuid = str(uuid.uuid3(uuid.NAMESPACE_DNS, 'ptah'))

# def use_list:
#     room_list = []
#     for


# while 1:
#     a = a + 1
