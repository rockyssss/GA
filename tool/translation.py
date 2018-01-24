# 区的平移
def translation(spacesarray, x, y):
    for space in spacesarray:
        # X对应的数组下标
        space[0] = space[0] + x
        # Y对应的数组下标
        space[1] = space[1] + y
    return spacesarray