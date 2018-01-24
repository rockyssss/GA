# _*_ coding: UTF-8 _*_


class Region:
    def __init__(self, name, type, length, width, space_list, point3d, Direction):
        """
        表示某够区域，其中包含区域名name，所属类型type，长度length，宽度width，
        空间列表space_list，参考起始点point3d，区域的方向Direction
        :param name: 中文名或者英文名
        :param type: 所属类别
        :param length: 沿着direction方向的长度
        :param width: 垂直于direction方向的长度
        :param space_list: 该区域空间的列表
        :param point3d: 该区域的参考起始点
        :param Direction: 该区域的方向
        """
        self.name = name
        self.type = type
        self.length = length  # 沿着direction方向的长度
        self.width = width  # 垂直于direction方向的长度
        self.space_list = space_list
        self.point3d = point3d
        self.Direction = Direction


class Direction:
    def __init__(self, dx, dy, x, y, z):
        """
        某个区域或者房间的 方向
        :param dx: 若不为零，表明该排列沿着x轴正（负）方向排列
        :param dy: 若不为零，表明该排列沿着y轴正（负）方向排列
        :param x: 若dx != 0，则x = dx，若dx = 0，则x的正负号表示其width沿着方向
        :param y: 若dy != 0，则y = dy，若dy = 0，则y的正负号表示其width沿着方向
        :param z: z方向表示高度
        """
        self.dx = dx
        self.dy = dy
        self.x = x
        self.y = y
        self.z = z


class Space:
    def __init__(self, name, type, rectangle):
        """
        某个房间的属性，其中包含房间名称、类型、矩形数据
        :param name:房间名称
        :param type:房间类型
        :param rectangle:房间的矩形数据（长、宽、参考起始点，方向）
        """
        self.name = name
        self.type = type
        self.rectangle = rectangle  # TODO:change to list test
        # self.point3D    #矩形对应的原点坐标


class Rectangle:
    def __init__(self, length, width, point3d, Direction, bound_list):
        """
        某个区域或者房间的矩形数据
        :param length: 矩形的长度（与direction方向相同）
        :param width: 矩形的宽度（与direction方向垂直）
        :param point3d: 矩形的三维坐标点（x,y,z）
        :param Direction: 矩形的方向
        :param bound_list: 存储rectangle中各个墙面的邻接情况
                            默认为走廊，为<0的数，数字绝对值越大，其存在的可能性概率越大
                            若邻接为房间的某个rectangle，则存储邻接房间在spaces的列表中的下标
        """
        self.length = length  # 沿着direction方向的长度
        self.width = width  # 垂直于direction方向的长度
        self.point3d = point3d
        self.Direction = Direction
        self.bound_list = bound_list  # bound_list存储


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

