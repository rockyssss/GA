from dao import rectangle_index as rec
# rectangle镜像
def image(spacesarray, index, coordinate):
    if (coordinate == 'x'):
        # Y对应数组的索引
        spacesarray[index][rec.Y_MIN_INDEX] = -spacesarray[index][rec.Y_MIN_INDEX]
        spacesarray[index][rec.Y_MAX_INDEX] = -spacesarray[index][rec.Y_MAX_INDEX]
    else:
        # X对应数组的索引
        spacesarray[index][rec.X_MIN_INDEX] = -spacesarray[index][rec.X_MIN_INDEX]
        spacesarray[index][rec.X_MAX_INDEX] = -spacesarray[index][rec.X_MAX_INDEX]
    return spacesarray

def imagespace(spacesarray, spacesdic, spaceid, coordinate):
    listrec = spacesdic.get(spaceid)
    for index in listrec:
        if (coordinate == 'x'):
            # Y对应数组的索引
            spacesarray[index][rec.Y_MIN_INDEX] = -spacesarray[index][rec.Y_MIN_INDEX]
            spacesarray[index][rec.Y_MAX_INDEX] = -spacesarray[index][rec.Y_MAX_INDEX]
        else:
            # X对应数组的索引
            spacesarray[index][rec.X_MIN_INDEX] = -spacesarray[index][rec.X_MIN_INDEX]
            spacesarray[index][rec.X_MAX_INDEX] = -spacesarray[index][rec.X_MAX_INDEX]
    return spacesarray