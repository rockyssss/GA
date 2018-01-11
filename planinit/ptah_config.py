# _*_ coding: UTF-8 _*_
import os

MONGO_URL = 'mongodb://192.168.0.209:27017/'
MONGO_USER = 'ptah'
MONGO_PASSWD = 'ptah'

DELIVERY_PATH = ""

if os.path.exists(os.getcwd() + "/../file/Delivery.csv"):
    DELIVERY_PATH = os.getcwd() + "/../file/Delivery.csv"  # 获取训练的数据
elif os.path.exists(os.getcwd() + "/file/Delivery.csv"):
    DELIVERY_PATH = os.getcwd() + "/file/Delivery.csv"  # 获取训练的数据

PROPERTIES_PATH = ""
if os.path.exists(os.getcwd() + "/../file/properties.csv"):
    PROPERTIES_PATH = os.getcwd() + "/../file/properties.csv"  # 获取训练的数据
elif os.path.exists(os.getcwd() + "/file/properties.csv"):
    PROPERTIES_PATH = os.getcwd() + "/file/properties.csv"  # 获取训练的数据

RATIO_PATH = ""
if os.path.exists(os.getcwd() + "/../file/ratio.csv"):
    RATIO_PATH = os.getcwd() + "/../file/ratio.csv"  # 获取训练的数据
elif os.path.exists(os.getcwd() + "/file/ratio.csv"):
    RATIO_PATH = os.getcwd() + "/file/ratio.csv"  # 获取训练的数据
