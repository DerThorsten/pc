from __future__ import print_function
from __future__ import division

import functools
import traceback
import numpy
from concurrent.futures import ThreadPoolExecutor

from threadpool import *
import time
import pylab
import vigra
import os
import colorama 
from colorama import Fore, Back, Style
colorama.init()


class Timer:
    def __init__(self, name, verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:

            print(Back.GREEN+Fore.RED+self.name+"..."+Style.RESET_ALL) 
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.verbose  :
            print(Back.GREEN+Fore.RED+self.name+"... took "+Style.RESET_ALL+str(self.interval)+"sec") 





def getSlicing(begin, end):
    return [slice(b,e) for b,e in zip(begin,end)]

def addHalo(shape, blockBegin, blockEnd, halo):

    withHaloBlockBegin = (
        max(blockBegin[0] - halo[0],0)   , 
        max(blockBegin[1] - halo[1],0)   ,
        max(blockBegin[2] - halo[2],0)
    )

    withHaloBlockEnd = (
        min(blockEnd[0] + halo[0],shape[0])   , 
        min(blockEnd[1] + halo[1],shape[1])   ,
        min(blockEnd[2] + halo[2],shape[2])
    )


    inBlockBegin = (
        blockBegin[0] -  withHaloBlockBegin[0],
        blockBegin[1] -  withHaloBlockBegin[1],
        blockBegin[2] -  withHaloBlockBegin[2]
    )

    inBlockEnd = (
        inBlockBegin[0] +  (blockEnd[0] - blockBegin[0]),
        inBlockBegin[1] +  (blockEnd[1] - blockBegin[1]),
        inBlockBegin[2] +  (blockEnd[2] - blockBegin[2])
    )

    return  withHaloBlockBegin, withHaloBlockEnd, inBlockBegin, inBlockEnd



def forEachBlock(shape, blockShape, f, nWorker, roiBegin=None, roiEnd=None):
    if roiBegin is None:
        roiBegin = (0,0,0)

    if roiEnd is None:
        roiEnd = shape

    if nWorker == 1:
        for blockIndex, blockBegin, blockEnd in blockYielder(roiBegin, roiEnd, blockShape):
            f(blockIndex=blockIndex, blockBegin=blockBegin, blockEnd=blockEnd)

    else:

        if False:
            futures = []
            with ThreadPoolExecutor(max_workers=nWorker) as executer:
                for blockIndex, blockBegin, blockEnd in blockYielder(roiBegin, roiEnd, blockShape):
                    executer.submit(f, blockIndex=blockIndex,blockBegin=blockBegin, blockEnd=blockEnd)

            for future in futures:
                e = future.exception()
                if e is not None:
                    raise e


        if True:
            pool = ThreadPool(nWorker)
            for blockIndex, blockBegin, blockEnd in blockYielder(roiBegin, roiEnd, blockShape):
                pool.add_task(f, blockIndex=blockIndex,blockBegin=blockBegin, blockEnd=blockEnd)
              
            pool.wait_completion()




def getShape(dataH5Dsets, labelsH5Dset=None):

    shape = None
    if labelsH5Dset is not None:
        shape = tuple(labelsH5Dset.shape[0:3])

    for inputFileName in dataH5Dsets.keys():
        fshape = tuple(dataH5Dsets[inputFileName].shape[0:3])
        if shape is not None:
            if(shape != fshape):
                raise RuntimeError("%s !- %s"%(str(shape),str(fshape)))
            assert shape == fshape
        else:
            shape = fshape
            
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
    
    blockIndex = 0 
    for xBegin in range(begin[0], end[0], blockShape[0]):
        xEnd = min(xBegin + blockShape[0],end[0])
        for yBegin in range(begin[1], end[1], blockShape[1]):
            yEnd = min(yBegin + blockShape[1],end[1])
            for zBegin in range(begin[2], end[2], blockShape[2]):
                zEnd =  min(zBegin + blockShape[2],end[2])


                yield blockIndex, (xBegin, yBegin, zBegin), (xEnd, yEnd, zEnd)
                blockIndex += 1




def labelsBoundingBox(labels, blockBegin, blockEnd):
    whereLabels = numpy.array(numpy.where(labels!=0))

    inBlockBegin = numpy.min(whereLabels,axis=1)
    inBlockEnd   = numpy.max(whereLabels,axis=1) +1
    
    subBlockShape = [e-b for e,b in zip(inBlockEnd, inBlockBegin)]



    labelsBlock = labels[inBlockBegin[0]:inBlockEnd[0], 
                         inBlockBegin[1]:inBlockEnd[1], 
                         inBlockBegin[2]:inBlockEnd[2]]

    globalBlockBegin = (
        blockBegin[0] + inBlockBegin[0], 
        blockBegin[1] + inBlockBegin[1],
        blockBegin[2] + inBlockBegin[2]
    )

    globalBlockEnd = (
        blockBegin[0] + inBlockBegin[0]+subBlockShape[0], 
        blockBegin[1] + inBlockBegin[1]+subBlockShape[1],
        blockBegin[2] + inBlockBegin[2]+subBlockShape[2]
    )

    whereLabels[0,:] -= inBlockBegin[0]
    whereLabels[1,:] -= inBlockBegin[1]
    whereLabels[2,:] -= inBlockBegin[2]

    return labelsBlock, globalBlockBegin, globalBlockEnd, whereLabels