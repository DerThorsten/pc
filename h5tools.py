def closeH5(h5File):
    try:
        h5File.close()
    except:
        pass


def closeAllH5Files(h5Files):
    for h5File in h5Files:
        closeH5(h5File)


def loadLabelsBlock(h5Dset, begin, end):
    return h5Dset[begin[0]:end[0], begin[1]:end[1], begin[2]:end[2], 0]