# _*_ coding: UTF-8 _*_
import os
import shelve
import pgmpy



def get_cpd(p, cpd):
    """
    :param p: 需求数据，如p = {'fruit': 'banana', 'size': 'small'}
    :param cpd: 对应需求的cpd表，如mle.estimate_cpd('tasty')
    :return: 邻接情况概率字典或列表
    示例如下：
    save_path = os.getcwd()+'\\hmh\\hmh1.txt' #传入cpd表对应的shelve文件的路径
    with shelve.open(save_path) as f:
        f['cpd'] = cpd
    p = {'51172100':0,'51442115':1}
    get_cpd(p, cpd)
    输出为：
    [('type', '51442115', 0.29999999999999999), ('type', '51442113', 0.29999999999999999), ('type', '51172100', 0.29999999999999999), ('type', '51445500', 0.10000000000000001), ('type', '51578800', 0.0), ('type', '51543115', 0.0), ('type', '51443315', 0.0), {'51172100': 0, '51442113': 0, '51442115': 1, '51443315': 0, '51445500': 0, '51543115': 0, '51578800': 0}]
    其中前len(var)所有可能连接项概率依次递减列表，最后项为条件概率项
    """
    cpd_result = {}
    for i in range(cpd.variable_card):
        list = []
        # cpd.variables为各列数据集合
        # cpd.variables第一个数据为表中测量项名，第二个为第1个变量控制维度名称，第三个为第2个变量控制维度名称，以此类推，
        # 如其为cpd.variables = ['tasty', 'fruit', 'size']，item为其中某个值，如item = 'fruit'
        for index, item in enumerate(cpd.variables):
            if item in p.keys():
                # print(p)
                # 如cpd.state_names = {'fruit': ['apple', 'banana'], 'size': ['large', 'small'], 'tasty': ['no', 'yes']}
                # 因此c为获取p中key为item的值在cpd.state_names中对应列表中索引值
                # 若item = 'fruit'，p[item]='banana',找对其对应cpd.state_names[item]=['apple', 'banana']的索引c为1
                c = cpd.state_names[item].index(p[item])
                # list[index-1] = c
                list.append(c)
            elif item is not cpd.variables[0]:
                list.append(0)
        b = cpd.values[i]
        # list存储每个变量的下表，循环获取变量下标，以获取需要提取的变量值
        for x in list:
            b = b[x]
        # d = cpd.values[i][list[0]][list[1]]
        # print(b)
        bro = cpd.variable
        val = cpd.state_names[cpd.variable][i]
        cpd_result[(bro, val)] = b
    sort_keys = sorted(cpd_result, key=lambda x: cpd_result[x])[::-1]
    sort_result_list = []
    for j in sort_keys:
        sort_result_list.append((j[0], j[1], cpd_result[j]))
    sort_result_list.append({cpd.variables[index + 1]: x for index, x in enumerate(list)})
    # print(sort_result_list)
    return sort_result_list


def Cpd(path):
    save_path = os.getcwd() + path  # 传入cpd表对应的shelve文件的路径
    with shelve.open(save_path) as f:
        cpd = f.get('cpd')
    return cpd


# p = {'51442115': 1}
# dict = {'51578800': 0, '51172100': 1, '51445500': 2, '51443315': 3, '51442115': 4, '51442113': 5, '51543115': 6}
# list1 = [2, 2, 3, 1, 2, 2, 1]
# list = get_cpd(p, Cpd("\\hmh1.txt"))
# # print(list[0][1])
# 获取墙的邻接关系和解决腔的个数问题
def Array(dic, list, list1, NewList):
    value = []
    i = 0
    j = 0
    while (i < len(list) - 1):
        a = list[i]
        b = a[1]
        c = NewList
        d = list[i][1]
        if (list[i][1] not in NewList):
            ll = list[i][1]
            dd = dic.get(list[i][1])
            if (list1[dic.get(list[i][1])] > 1):
                NewList.append(list[i][1])
                while (j < list1[dic.get(list[i][1])]):
                    value.append(list[i][1])
                    # value.append(list[i][2])
                    j += 1
                break
            else:
                NewList.append(list[i][1])
                value.append(list[i][1])
                # value.append(list[i][2])
                break
        else:
            i += 1
    return value


def loop(pset, dic, list, list1, path):
    p = pset
    i = 0
    subscript = []
    NewList = []
    while i < len(list) - 1:
        lists = get_cpd(p, Cpd(path))
        value = Array(dic, lists, list1, NewList)
        p = {value[0]: 1}
        for j in value:
            subscript.append(j)
        i += 1
    return subscript

# values = loop(p, dict, list, list1, "\\hmh1.txt")
# print(values)
# for i in values:
#     print(dict.get(i))
