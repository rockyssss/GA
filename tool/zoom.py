from dao import rectangle_index as rec
# rectangle缩放
def zoom(spacesarray, index, ratio):
    # length对应数组下标索引
    spacesarray[index][rec.L_INDEX] = spacesarray[index][rec.L_INDEX] * ratio
    # width对应数组下标索引
    spacesarray[index][rec.W_INDEX] = spacesarray[index][rec.W_INDEX] * ratio
    return spacesarray

# space缩放

def zoomspace(spacesarray, spacesdic, spaceid, ratio):
    listrec = spacesdic.get(spaceid)
    for index in listrec:
        spacesarray[index][rec.L_INDEX] = spacesarray[index][rec.L_INDEX] * ratio
        # Y对应的数组下标
        spacesarray[index][rec.W_INDEX] = spacesarray[index][rec.W_INDEX] * ratio
    return spacesarray

