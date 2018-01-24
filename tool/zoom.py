# 缩放
def zoom(spacesarray, ratio):
    for space in spacesarray:
        # length对应数组下标索引
        space[4] = space[4] * ratio
        # width对应数组下标索引
        space[5] = space[4] * ratio
    return spacesarray