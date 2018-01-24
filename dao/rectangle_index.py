# _*_ coding: UTF-8 _*_
"""
示例:

a = np.array([0, 1, 2, 7, 8, 3, 4, 5, 2, -25, -50, -25,2])
若a = np.array([X_MIN, Y_MIN, Xm, Ym, L, W, SPACE_ID, TYPE, B1, B2, B3, B4, BOUND])
此时则有：
a[X_MIN_INDEX]为X------------------矩形a 左下角点x坐标(最小的x)为X
a[Y_MIN_INDEX]为Y------------------矩形a 左下角点y坐标(最小的y)为Y
a[X_MAX_INDEX]为X_MAX--------------矩形a 左下角点x坐标(最大的x)为X_MAX
a[Y_MAX_INDEX]为Y_MAX--------------矩形a 左下角点y坐标(最大的y)为Y_MAX
a[L_INDEX]为L----------------------矩形a 长度为L
a[W_INDEX]为W----------------------矩形a 宽度为W
a[SPACE_ID_INDEX]为SPACE_ID--------矩形a 对应的space_id为SPACE_ID
a[TYPE_INDEX]为TYPE----------------矩形a 对应的type为TYPE
a[B1_INDEX]为B1--------------------矩形a 若B1>0则B1边邻接的房间为B1号房间，否则若B1<0则B1边邻接走廊的概率为B1
a[B2_INDEX]为B2--------------------矩形a 若B2>0则B1边邻接的房间为B1号房间，否则若B1<0则B1边邻接走廊的概率为B2
a[B3_INDEX]为B3--------------------矩形a 若B3>0则B1边邻接的房间为B1号房间，否则若B1<0则B1边邻接走廊的概率为B3
a[B4_INDEX]为B4--------------------矩形a 若B4>0则B1边邻接的房间为B1号房间，否则若B1<0则B1边邻接走廊的概率为B4
a[BOUND_INDEX]为BOUND--------------矩形a 所在的区域边界的索引为BOUND

"""
RS_LEN=13
X_MIN_INDEX = 0
Y_MIN_INDEX = 1
X_MAX_INDEX = 2
Y_MAX_INDEX = 3
L_INDEX = 4
W_INDEX = 5
SPACE_ID_INDEX = 6
TYPE_INDEX = 7
B1_INDEX = 8
B2_INDEX = 9
B3_INDEX = 10
B4_INDEX = 11
BOUND_INDEX = 12

Aname = ['麻醉准备间 ', '麻醉恢复区 ', '精密仪器室 ', '无菌包储存室 ', '一次性物品储存室 ', '体外循环室 ', '麻醉用品库 ', '总护士站 ', '快速灭菌', '腔镜洗消间',
         '家属等候区 ',
         '患者换床间 ', '转运床停放间 ', '废弃物分类暂存间', '器械预清洗间 ', '医疗气体制备间 ', '石膏制备间 ', '病理标本间 ', '保洁用品存放间 ', '衣物发放及入口管理 ',
         '更衣 ', '换鞋间',
         '淋浴间', '卫生间', '二次换鞋间 ', '护士长办公室 ', '麻醉师办公室 ', '示教室 ', '休息就餐间 ', '卫生员室 ', '库房', '会诊室', '电梯', '楼梯', '洁净走廊',
         '清洁走廊',
         '管井', '前室', '前厅', '缓冲', '手术药品库 ', '保洁室 ', '谈话室', '污染被服暂存间 ', '值班室 (部分含卫生间)', '缓冲走廊', '麻醉科主任办公室 ', '一级手术室',
         '多功能复合手术室', '二级手术室', '三级手术室', '正负压转换手术室']  # 数据的列名称集合

Atype = [51441115, 51443500, 51442113, 51172100, 51171300, 51348400, 51441100, 51172311, 51578800, 51442115, 51443300,
         51443315, 51573915, 51579200, 51445500, 51578600, 51578000, 51671813, 63138000, 51444582, 51444500, 51444580,
         65131100, 65131300, 51444511, 51172382, 51178400, 51178000, 51443911, 51178600, 51442123, 51348000, 23111111,
         23111300, 25118200, 25118400, 23112500, 25131700, 25131900, 51442141, 51543115, 63130000, 51171700, 51172500,
         51172384, 25118700, 51441111, 51446100, 51448700, 51446300, 51446500, 51449600]
Adict = {}
for index in range(len(Aname)):
    Adict[Atype[index]] = Aname[index]


