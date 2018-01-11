# _*_ coding: UTF-8 _*_

"""
    获取区域
    的为整形,并且是0.3的倍数
"""
def widthInt(basic):
    tmp = int(basic/0.3)
    while (not (tmp * 0.3 % 1 == 0)):
        tmp -= 1
    return tmp * 0.3

def lengthInt(basic):
    tmp = int(basic/0.3)
    while (not (tmp * 0.3 % 1 == 0)):
        tmp += 1
    return tmp * 0.3
#mton
def getm2nInt(y, m, n):
    #mx*nx=y  sq为x
    sq = (y/(m*n)) ** 0.5
    # width = int(sq / 0.3) * 0.3
    # length = int((y / width + 1) / 0.3) * 0.3
    # #长宽都为整数的情况
    # 1:1情
    bx=m*sq
    by=n*sq
    x1=0
    y1=0
    if(m>=n):
        x1 = lengthInt(bx)
        y1 = widthInt(by)
    else:
        x1 = widthInt(bx)
        y1 = lengthInt(by)
    end1=x1*y1

    x2=x1
    y2=lengthInt(y/x2)
    end2=x2*y2

    if(abs(end1-y)<abs(end2-y)):
        return [x1,y1]
    else:
        return [x2, y2]


"""
    获取房间级别的float代码
"""


def getm2n(y, m, n):
    # mx*nx=y  sq为x
    sq = (y / (m * n)) ** 0.5
    # width = int(sq / 0.3) * 0.3
    # length = int((y / width + 1) / 0.3) * 0.3
    # #长宽都为整数的情况
    # 1:1情
    bx = m * sq
    by = n * sq
    return [bx, by]


    # y=3000
    # #外形边界,要求xy都应该是300mm的整数倍
    # a=getm2n(y,1,1)
    # print(a[0]*a[1])
    # b=getm2n(y,2,1)
    # print(b[0]*b[1])
    # c=getm2n(y,3,2)
    # print(c[0]*c[1])
    # d=getm2n(y,2,3)
    # print(d[0]*d[1])
    # e=getm2n(y,2,7)
    # print(e[0]*e[1])
    #
    #
    # print()
