#!/usr/bin/python
#_*_ coding: UTF-8 _*_
import csv

"""
    日常数据分析时最常打交道的是csv文件和list,dict类型。涉及到的主要需求有：

    将一个二重列表[[],[]]写入到csv文件中
    从文本文件中读取返回为列表
    将一字典写入到csv文件中
    从csv文件中读取一个字典
    从csv文件中读取一个计数字典
"""

# 功能：将一个二重列表写入到csv文件中
# 输入：文件名称，数据列表
def createListCSV(fileName="", dataList=[]):
    with open(fileName, "wb") as csvFile:
        csvWriter = csv.writer(csvFile)
        for data in dataList:
            csvWriter.writerow(data)
        csvFile.close

# 功能：从文本文件中读取返回为列表的形式
# 输入：文件名称，分隔符（默认,）
def readListCSV(fileName="", splitsymbol=","):
    dataList = []
    with open(fileName, "r") as csvFile:
        # dataLine = csvFile.readline().strip("\n")
        dataLine = csvFile.readline().strip("\n")
        while dataLine != "":
            tmpList = dataLine.split(splitsymbol)
            if(tmpList[-1]==""):
                dataList.append(tmpList[:-1])
            else:
                dataList.append(tmpList)
            dataLine = csvFile.readline().strip("\n")
        csvFile.close()
    return dataList

# 功能：将一字典写入到csv文件中
# 输入：文件名称，数据字典
def createDictCSV(fileName="", dataDict={}):
    with open(fileName, "wb") as csvFile:
        csvWriter = csv.writer(csvFile)
        for k,v in dataDict.iteritems():
            csvWriter.writerow([k,v])
        csvFile.close()


# 功能：从csv文件中读取一个字典
# 输入：文件名称，keyIndex,valueIndex
def readDictCSV(fileName="", keyIndex=0, valueIndex=1):
    dataDict = {}
    with open(fileName, "r") as csvFile:
        #读取第一行
        dataLine = csvFile.readline().strip("\n")
        while dataLine != "":
            tmpList = dataLine.split(",")
            dataDict[tmpList[keyIndex]] = tmpList[valueIndex]
            dataLine = csvFile.readline().strip("\n")
        csvFile.close()
    return dataDict

# dict1=readListCSV("resultAreaT1.csv")
# print()