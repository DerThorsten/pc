from __future__ import print_function
from __future__ import division

import functools
import traceback
import numpy
from concurrent.futures import ThreadPoolExecutor



def forEachBlock(shape, blockShape, f, nWorker):

    futures = []
    with ThreadPoolExecutor(max_workers=nWorker) as executer:
        for blockBegin, blockEnd in blockYielder((0,0,0), shape, blockShape):
            if nWorker == 1:
                f(blockBegin=blockBegin, blockEnd=blockEnd)
            else:
                future = executer.submit(f, blockBegin=blockBegin, blockEnd=blockEnd)
                futures.append(future)



    for future in futures:
        e = future.exception()
        if e is not None:
            raise e









def getShape(dataH5Dsets, labelsH5Dset):
    shape = tuple(labelsH5Dset.shape[0:3])
    for inputFileName in dataH5Dsets.keys():
        fshape = tuple(dataH5Dsets[inputFileName].shape[0:3])
        assert shape == fshape
    return shape


def reraise_with_stack(func):

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback_str = traceback.format_exc(e)
            raise StandardError("Error occurred. Original traceback "
                                "is\n%s\n" % traceback_str)

    return wrapped



def blockYielder(begin, end, blockShape):
    

    for xBegin in range(begin[0], end[0], blockShape[0]):
        xEnd = min(xBegin + blockShape[0],end[0])
        for yBegin in range(begin[1], end[1], blockShape[1]):
            yEnd = min(yBegin + blockShape[1],end[1])
            for zBegin in range(begin[2], end[2], blockShape[2]):
                zEnd =  min(zBegin + blockShape[2],end[2])

                yield (xBegin, yBegin, zBegin), (xEnd, yEnd, zEnd)





def labelsBoundingBox(labels, blockBegin, blockEnd):
    whereLabels = numpy.array(numpy.where(labels!=0))

    inBlockBegin = numpy.min(whereLabels,axis=1)
    inBlockEnd   = numpy.max(whereLabels,axis=1) +1
    #print(inBlockBegin,inBlockEnd,inBlockEnd-inBlockBegin)



    labelsBlock = labels[inBlockBegin[0]:inBlockEnd[0], 
                         inBlockBegin[1]:inBlockEnd[1], 
                         inBlockBegin[2]:inBlockEnd[2]]

    globalBlockBegin = (
        blockBegin[0] + inBlockBegin[0], 
        blockBegin[1] + inBlockBegin[1],
        blockBegin[2] + inBlockBegin[2]
    )

    globalBlockEnd = (
        blockEnd[0] + inBlockBegin[0], 
        blockEnd[1] + inBlockBegin[1],
        blockEnd[2] + inBlockBegin[2]
    )
    #print("whereLabelsShape",whereLabels)

    whereLabels[0,:] -= inBlockBegin[0]
    whereLabels[1,:] -= inBlockBegin[1]
    whereLabels[2,:] -= inBlockBegin[2]

    return labelsBlock, globalBlockBegin, globalBlockEnd, whereLabels