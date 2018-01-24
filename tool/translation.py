from dao import rectangle_index as rec

# rectangle的平移
def translation(spacesarray, index, delta_x, delta_y):
        # X对应的数组下标
    spacesarray[index][rec.X_MIN_INDEX] = spacesarray[index][rec.X_MIN_INDEX] + delta_x
    # Y对应的数组下标
    spacesarray[index][rec.Y_MIN_INDEX] = spacesarray[index][rec.Y_MIN_INDEX] + delta_y
    return spacesarray


# space的平移
def tanslationrec(spacesarray, spacesdic, spaceid, delta_x, delta_y):
    listrec = spacesdic.get(spaceid)
    for index in listrec:
        spacesarray[index][rec.X_MIN_INDEX] = spacesarray[index][rec.X_MIN_INDEX] + delta_x
        # Y对应的数组下标
        spacesarray[index][rec.Y_MIN_INDEX] = spacesarray[index][rec.Y_MIN_INDEX] + delta_y
    return spacesarray