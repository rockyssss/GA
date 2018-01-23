# _*_ coding: UTF-8 _*_

import dao.spaceApoint
from tool import dao_numpy,draw

space_list = []

space_list.append(
    dao.spaceApoint.Space("Room1", 51443500, [dao.spaceApoint.Rectangle(10, 20, dao.spaceApoint.Point3D(0, 0, 0),
                                                                   dao.spaceApoint.Direction(1, 0, 1, 1, 0),1,2,3,4)]))
space_list.append(
    dao.spaceApoint.Space("Room2", 51443500, [dao.spaceApoint.Rectangle(10, 20, dao.spaceApoint.Point3D(10, 0, 0),
                                                                   dao.spaceApoint.Direction(1, 0, 1, 1, 0),1,2,3,4)]))
space_list.append(
    dao.spaceApoint.Space("Room2", 51443500, [dao.spaceApoint.Rectangle(10, 20, dao.spaceApoint.Point3D(20, 0, 0),
                                                                   dao.spaceApoint.Direction(1, 0, 1, 1, 0),1,2,3,4)]))
space_list.append(
    dao.spaceApoint.Space("Room3", 51443500, [dao.spaceApoint.Rectangle(30, 10, dao.spaceApoint.Point3D(0, 20, 0),
                                                                   dao.spaceApoint.Direction(1, 0, 1, 1, 0),1,2,3,4)]))
space_list.append(
    dao.spaceApoint.Space("Room3", 51443500, [dao.spaceApoint.Rectangle(20, 10, dao.spaceApoint.Point3D(0, 20, 0),
                                                                   dao.spaceApoint.Direction(1, 0, 1, 1, 0),1,2,3,4)]))

mynp=dao_numpy.dao2np(space_list)

draw.draw(mynp)