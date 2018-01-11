# _*_ coding: UTF-8 _*_

import numpy as np
from robust_regression.rb import RobustLinearRegression
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor
import matplotlib.pyplot as plt
from numpy import genfromtxt
import csv
import os
import math
from planinit.ptah_config import *

Number = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0,
          1, 0, 0, 1, 1, 2, 1, 3, 1, 1]


def LinearCoef_2():
    list_result2 = []
    i = 1
    rng = np.random.RandomState(0)
    n_features = 1
    data_path = PROPERTIES_PATH
    # X, y = make_regression(n_samples=20, n_features=n_features, random_state=0, noise=4.0,bias=100.0)


    delivery_data = genfromtxt(data_path, delimiter=',')
    while i < 41:
        # 将数据赋值给要训练的X，Y(因变量，自变量)
        X = delivery_data[:, -6: -5]
        y = delivery_data[:, 0]
        X_outliers = rng.normal(0, 0.5, size=(4, n_features))
        y_outliers = rng.normal(0, 2.0, size=4)
        X_outliers[:2, :] += X.max() + X.mean()
        X_outliers[2:, :] += X.min() - X.mean()
        y_outliers[:2] += y.min() - y.mean()
        y_outliers[2:] += y.max() * 2 + y.mean()
        X = np.vstack((X, X_outliers))
        y = np.concatenate((y, y_outliers))
        robust_regression_estimator = RobustLinearRegression()
        huber_estimator = HuberRegressor()
        robust_regression_estimator.fit(X, y)
        huber_estimator.fit(X, y)
        list_result2.extend(huber_estimator.coef_)
        list_result2.append(huber_estimator.intercept_)
        i += 1
    return list_result2


def LinearCoef_1():
    listresult1 = []
    i = 1
    rng = np.random.RandomState(0)
    n_features = 1
    # X, y = make_regression(n_samples=20, n_features=n_features, random_state=0, noise=4.0,bias=100.0)

    data_path = DELIVERY_PATH

    delivery_data = genfromtxt(data_path, delimiter=',')
    while (i < 6):
        # 将数据赋值给要训练的X，Y(因变量，自变量)
        X = delivery_data[:, -(i + 1): -i]
        y = delivery_data[:, 0]
        x_outliers = rng.normal(0, 0.5, size=(4, n_features))
        y_outliers = rng.normal(0, 2.0, size=4)
        x_outliers[:2, :] += X.max() + X.mean()
        x_outliers[2:, :] += X.min() - X.mean()
        y_outliers[:2] += y.min() - y.mean()
        y_outliers[2:] += y.max() * 2 + y.mean()
        X = np.vstack((X, x_outliers))
        y = np.concatenate((y, y_outliers))

        huber_estimator = HuberRegressor()
        # robust_regression_estimator.fit(X, y)
        huber_estimator.fit(X, y)
        # assert (huber_estimator.coef_ * robust_regression_estimator.coef_[:-1] > 0).all(
        listresult1.extend(huber_estimator.coef_)
        listresult1.append(huber_estimator.intercept_)
        i += 1
    return listresult1


# result1 = LinearCoef_1()
# result2 = LinearCoef_2()

def SaveAsCsv(ListResult, path):
    i = 0
    if (os.path.exists(path)):
        os.remove(path)
    with open(path, 'a+', newline='') as data:
        writer = csv.writer(data, dialect='excel')
        while (i < len(ListResult)):
            # writer.writerow(title)
            writer.writerow([ListResult[i], ListResult[i + 1]])
            i += 2


def ReadCsv(path):
    with open(path) as f:
        datareader = csv.reader(f);
        list1 = list(datareader)
    return list1


def Calculate(ListPagrame, Area):
    y = Area
    x = 0.0
    count = 0
    AreaArea = []
    while (count < len(ListPagrame)):
        a = ListPagrame[count]
        b = ListPagrame[count + 1]
        x = (y - b) / a
        AreaArea.append(x)
        count += 2
    return AreaArea


def Calculate1(ListPagrame, number):
    x = number
    y = 0.0
    count = 0
    AreaArea = []
    while (count < len(ListPagrame)):
        a = ListPagrame[count]
        b = ListPagrame[count + 1]
        y = a * x + b
        AreaArea.append(y)
        count += 2
    return AreaArea


def test_negative_incline():
    ListResult3 = []
    rng = np.random.RandomState(0)
    n_features = 1
    dataPath = DELIVERY_PATH
    deliveryData = genfromtxt(dataPath, delimiter=',')
    i = 1
    while (i < 6):
        # 将数据赋值给要训练的X，Y(因变量，自变量)
        X = deliveryData[:, -(i + 1): -i]
        y = deliveryData[:, 0]
        X_outliers = rng.normal(0, 0.5, size=(4, n_features))
        y_outliers = rng.normal(0, 2.0, size=4)
        X_outliers[:2, :] += X.max() + X.mean()
        X_outliers[2:, :] += X.min() - X.mean()
        y_outliers[:2] += y.min() - y.mean()
        y_outliers[2:] += y.max() * 2 + y.mean()
        X = np.vstack((X, X_outliers))
        X = X.max() - X
        y = np.concatenate((y, y_outliers))
        huber_estimator = HuberRegressor()
        huber_estimator.fit(X, y)
        ListResult3.extend(huber_estimator.coef_)
        ListResult3.append(huber_estimator.intercept_)
        i += 1
    return ListResult3


def test_negative_incline_1():
    ListResult4 = []
    rng = np.random.RandomState(0)
    n_features = 1
    data_path = PROPERTIES_PATH  # 获取训练的数据
    # X, y = make_regression(n_samples=20, n_features=n_features, random_state=0, noise=4.0,bias=100.0)
    delivery_data = genfromtxt(data_path, delimiter=',')
    i = 1
    while i < 41:
        # 将数据赋值给要训练的X，Y(因变量，自变量)
        X = delivery_data[:, -(i + 1): -i]
        y = delivery_data[:, 0]
        X_outliers = rng.normal(0, 0.5, size=(4, n_features))
        y_outliers = rng.normal(0, 2.0, size=4)
        X_outliers[:2, :] += X.max() + X.mean()
        X_outliers[2:, :] += X.min() - X.mean()
        y_outliers[:2] += y.min() - y.mean()
        y_outliers[2:] += y.max() * 2 + y.mean()
        X = np.vstack((X, X_outliers))
        X = X.max() - X
        y = np.concatenate((y, y_outliers))
        # robust_regression_estimator = RobustLinearRegression()
        huber_estimator = HuberRegressor()
        # robust_regression_estimator.fit(X, y)
        huber_estimator.fit(X, y)
        ListResult4.extend(huber_estimator.coef_)
        ListResult4.append(huber_estimator.intercept_)
        i += 1
    return ListResult4


# result4 = test_negative_incline()
# result6 = test_negative_incline_1()

def RoundNumbers(ListResult1, ListResult2):
    count = 0
    l = ListResult1[0]
    while (count < len(ListResult1)):
        if (ListResult1[count] < 0):
            ListResult1[count] = ListResult2[count]
            ListResult1[count + 1] = ListResult2[count + 1]
        count += 2
    return ListResult1


# ll = RoundNumbers(result1, result4)
def WeightArea(list, i):
    count = 2
    if i < 4000:
        while (count <= len(list)):
            if (i <= 1000):
                list[count - 1] -= 300
                count += 2
            else:
                list[count - 1] += (i / 1000 - 2) * 300
                count += 2
        list[1] += (i / 1000) * 450
        list[3] += (i / 1000 - 1) * 1200
        list[5] += (i / 1000) * 300
        list[7] -= (i / 1000) * 50
    else:
        while (count <= len(list)):
            if (i <= 1000):
                list[count - 1] -= 300
                count += 2
            else:
                list[count - 1] += (i / 1000 - 2) * 300
                count += 2
        list[1] += (i / 1000) * 450
        list[3] += (i / 1000 - 1) * 800
        list[5] += (i / 1000) * 300
        list[7] -= (i / 1000) * 50
    return list


def OperationRoom(area):
    count = math.floor(area / 50)
    list = []
    a5 = 1
    a1 = math.ceil(count * 3 / 20)
    a2 = math.ceil(count * 10 / 20)
    a3 = math.ceil(count * 4 / 20)
    a4 = math.ceil(count * 2 / 20)
    list.append(a1)
    list.append(a5)
    list.append(a2)
    list.append(a3)
    list.append(a4)
    return list


def Weight(list, number):
    count = 0
    while (count < len(list)):
        list[count] *= 10
        count += 2
    list[0] *= 2
    list[2] /= (2 * number / 20)
    list[3] += (number - 10)
    list[4] /= (2 * number / 20)
    list[5] += (number - 10)
    list[6] /= (5 * number / 30)
    list[7] += (number - 10)
    list[8] /= (2 * number / 10)
    list[9] += (number - 10)
    list[12] /= (2 * number / 10)
    list[13] += (number - 10)
    list[14] /= (5 * number / 20)
    list[15] += (number - 10)
    list[18] /= (2 * number / 10)
    list[19] += (number - 10)
    list[20] /= (3 * number / 15)
    list[21] += (number - 10)
    list[22] /= (2 * number / 10)
    list[23] += (number - 10)
    list[24] /= (3 * number / 15)
    list[25] += (number - 10)
    list[28] /= (2 * number / 10)
    list[29] += (number - 10)
    list[32] /= (number / 10)
    list[33] += (number - 10)
    list[34] /= (number / 10)
    list[35] += (number - 10)
    list[36] /= (number / 10)
    list[37] += (number - 10)
    list[38] *= (number / 10)
    list[39] += (number / 10)
    list[40] /= (5 * number / 10)
    list[41] += (number - 10)
    list[42] /= (number / 10)
    list[43] += (number - 10)
    list[44] /= (number / 10)
    list[45] += (number - 10)
    list[50] /= (number / 10)
    list[51] += (number - 10)
    list[58] /= (number / 10)
    list[59] += (number - 10)
    list[62] /= (number / 10)
    list[63] += (number - 10)
    list[64] /= (10 * number / 15)
    list[65] += (number - 10)
    list[66] /= (10 * number / 15)
    list[67] += (number - 10)
    list[68] /= (10 * number / 15)
    list[69] += (number - 10)
    list[70] /= (10 * number / 15)
    list[71] += (number - 10)
    list[72] /= (3 * number / 15)
    list[73] += (number - 10)
    list[74] /= (3 * number / 15)
    list[75] += (number - 10)
    list[76] /= (3 * number / 15)
    list[77] += (number - 10)
    return list


def ClassArea(y, list):
    list1 = list[:-12]
    list3 = list[-12:]
    # 求和
    sum1 = 0
    for area in list1:
        sum1 += area
    sum3 = 0
    for area in list3:
        sum3 += area

    # 比值
    ratio = float(sum1) / (y - sum3)

    list2 = []
    for area in list1:
        nArea = area / ratio
        list2.append(nArea)

    sum1 = 0
    for area in list2:
        sum1 += area

    return list2 + list3


# res=ClassArea(3000, ResultArea)
# print(ClassArea(3000, ResultArea))
# SaveAsCsv(ResultArea, r"E:\area.csv")
def SumArea(list, sumarea):
    # 不改变的数组
    list3 = list[-1:]
    # 要改变的数组
    list1 = list[:-1]
    # 求和
    sum1 = 0
    for area in list1:
        sum1 += area
    sum3 = 0
    for area in list3:
        sum3 += area
    # 比值
    ratio = float(sum1) / (sumarea - sum3)
    list2 = []
    for area in list1:
        nArea = area / ratio
        list2.append(nArea)
    sum1 = 0
    for area in list2:
        sum1 += area
    # print(list2)
    # print(sum1)
    return list2 + list3
