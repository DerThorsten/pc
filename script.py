import commentjson as json
import h5py
import vigra
import threading

from concurrent.futures import ThreadPoolExecutor


labelsFile = ("/home/tbeier/hhes/explicit_labels_semantic2.h5","data")
rawDataFile = ("/home/tbeier/hhes/2x2x2nm_chunked/data_normalized.h5","data")


blockSize = (64,64,64)


def blockYielder(begin, end, blockSize):
    
    for xBegin in range(begin[0], end[0], blockSize[0]):
        xEnd = xBegin + blockSize[0]
        for yBegin in range(begin[1], end[1], blockSize[1]):
            yEnd = yBegin + blockSize[1]
            for zBegin in range(begin[2], end[2], blockSize[2]):
                zEnd = zBegin + blockSize[2]

                yield (xBegin, yBegin, zBegin), (xEnd, yEnd, zEnd)


# load the meta data

labelsH5 = h5py.File(labelsFile[0],'r')[labelsFile[1]]
rawDataH5 = h5py.File(rawDataFile[0],'r')[rawDataFile[1]]

# spatial shape
shape =  rawDataH5.shape
print "shape", shape




# check for labels block shape
#labelsBlockShape = (500, 500 ,500)
#
#for blockBegin, blockEnd in blockYielder((0,0,0), shape, labelsBlockShape):
#    
#    labels = labelsH5[blockBegin[0]:blockEnd[0], blockBegin[1]:blockEnd[1], blockBegin[2]:blockEnd[2], 0]
#
#    print labels.max()
#
#
#
#sys.exit(0)

lock = threading.Lock()

# loop over all blocks
with ThreadPoolExecutor(max_workers=8) as e:


    for blockBegin, blockEnd in blockYielder( (1000,)*3,(2000,)*3, blockSize):

        def f(blockBegin, blockEnd):
            labels = labelsH5[blockBegin[0]:blockEnd[0], blockBegin[1]:blockEnd[1], blockBegin[2]:blockEnd[2], 0]
            if labels.max() > 0 :
                with lock:
                    print "braa"
        e.submit(f, blockBegin, blockEnd)
