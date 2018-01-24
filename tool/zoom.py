from dao import rectangle_index as rec
# 缩放
def zoom(spacesarray, ratio):
    for space in spacesarray:
        # length对应数组下标索引
        space[rec.L_INDEX] = space[rec.L_INDEX] * ratio
        # width对应数组下标索引
        space[rec.W_INDEX] = space[rec.W_INDEX] * ratio
    return spacesarray