import commentjson as json
import h5py
import vigra
import threading

from concurrent.futures import ThreadPoolExecutor



def checkShape(dataH5Dsets, labelsH5Dset):
    shape = tuple(labelsH5Dset.shape[0:3])
    for inputFileName in dataH5Dsets.keys():
        fshape = tuple(dataH5Dsets[inputFileName].shape[0:3])
        assert shape == fshape


def blockYielder(begin, end, blockSize):
    
    for xBegin in range(begin[0], end[0], blockSize[0]):
        xEnd = xBegin + blockSize[0]
        for yBegin in range(begin[1], end[1], blockSize[1]):
            yEnd = yBegin + blockSize[1]
            for zBegin in range(begin[2], end[2], blockSize[2]):
                zEnd = zBegin + blockSize[2]

                yield (xBegin, yBegin, zBegin), (xEnd, yEnd, zEnd)






