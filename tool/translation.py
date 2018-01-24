from dao import rectangle_index as rec
# 区的平移
def translation(spacesarray, x, y):
    for space in spacesarray:
        # X对应的数组下标
        space[rec.X_MIN_INDEX] = space[rec.X_MIN_INDEX] + x
        # Y对应的数组下标
        space[rec.Y_MIN_INDEX] = space[rec.Y_MIN_INDEX] + y
    return spacesarray