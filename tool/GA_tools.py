def isnumber(a_string):
    """
    判断字符串是否可以转为float型
    :param a_string:
    :return:
    """
    try:
        float(a_string)
        return True
    except:
        return False