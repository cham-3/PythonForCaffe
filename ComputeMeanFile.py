#/usr/bin python
#encoding=utf-8

import os
import numpy as np
import caffe
import sys
import matplotlib.pyplot as plt
import skimage

def computeMean():
  dirpath = "" # modify the image directory in your case
    mean_data = np.zeros([128,128], dtype=np.float) # modify the size of the image you used
    dirs = os.listdir(dirpath)
    dirs.sort()
    cnt = 0
    for d in dirs:
        idpath = os.path.join(dirpath, d)
        imgs = os.listdir(idpath)
        imgs.sort()
        print "Processing {}-th id".format(d)
        for img in imgs:
            imgpath = os.path.join(idpath, img)
            img = skimage.io.imread(imgpath, as_gray=True)
            data_mean += img
            cnt+=1
    mean_data = mean_data/cnt
    return mean_data

def npy2binaryproto(data):
    try:
        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.num=1
        blob.channels=1
        blob.height = data.shape[0]
        blob.width = data.shape[1]
        blob.data.extend(data.astype(float).flat)
        binaryproto_file = open('mean.binaryproto', 'wb' ) 
        binaryproto_file.write(blob.SerializeToString())
        binaryproto_file.close()
    except Exception,e:
        print "Error occurred! {}".format(e)

def binaryproto2npy(binary):
    try:
        blob = caffe.proto.caffe_pb2.BlobProto()
        data = open(binary , 'rb' ).read()
        blob.ParseFromString(data)
        arr = np.array( caffe.io.blobproto_to_array(blob) )
        out = arr[0]
        return arr
        #np.save( 'mean.npy' , out)
    except Exception,e:
        print "Error occurred! {}".format(e)

if __name__ == "__main__":
    print "Computing the mean of all the training images..."
    mean_data = computeMean()
    npy2binaryproto(mean_data)
    print "Done."
