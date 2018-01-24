from dao import rectangle_index as rec
# 镜像
def image(spacesarray, coordinate):
    for space in spacesarray:
        if (coordinate == 'x'):
            # Y对应数组的索引
            space[rec.Y_MIN_INDEX] = -space[rec.Y_MIN_INDEX]
        else:
            # X对应数组的索引
            space[rec.X_MIN_INDEX] = -space[rec.X_MIN_INDEX]
    return spacesarray