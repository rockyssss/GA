# 镜像
def image(spacesarray, coordinate):
    for space in spacesarray:
        if (coordinate == 'x'):
            # Y对应数组的索引
            space[1] = -space[1]
        else:
            # X对应数组的索引
            space[0] = -space[0]
    return spacesarray