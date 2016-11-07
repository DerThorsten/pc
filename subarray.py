import numpy




class Block(object):
    def __init__(self, begin, end):
        self.begin = begin
        self.end = end


class BlockWithBorder(object):



class SubArray(object):
    def __init__(self, shape, data=None, begin=None):
        self.shape = shape
        self.data = None
        self.begin = None
        self.end = None




class Helper(object):
    def __init__(self, dset):
        self.dset = dset
        self.shape = shape

    def getBlockWithHalo(self, innerBegin, innerEnd, halo):

        shape = self.shape

        outerBegin = [max(ib - h, 0) for ib, h in zip(innerBegin, halo)]
        outerEnd = [min(ie + h, s) for ie, h, s in zip(innerEnd, halo, shape)]

        innerBeginLocal    [ib-ob for ib, ob in zip(innerBegin, outerBegin)]
        innerBlockEndLocal [ie-ib for ie, ib in zip(innerEnd, innerBegin)]




if __name__ == "__main__":

    dset = numpy.ones((1000,1000,1000),dtype='uint8')


    h = Helper(dset)
    
    blockBegin = (10,  20, 30)
    blockEnd =   (110,110,110)
    halo = (20,20,20)


    h.getBlockWithHalo(blockBegin=blockBegin,blockEnd=blockEnd,
                       halo)


