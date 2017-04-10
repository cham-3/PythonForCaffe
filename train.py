#/usr/bin/env python
#encoding=utf-8
'''
@author:Chamsu
@institute:XMU
@date: 2017-03-27
'''

import sys
sys.path.insert(0,'/path/to/your/python/layer')
import caffe
import numpy as np
import glog as log


if __name__ == '__main__':
    caffe.set_mode_gpu()
    caffe.set_device(0)
    caffe.init_log()
    solver_file = '/path/to/your/solver/file'
    #pre_solver = '/path/to/your/pretrained/snapshot'
    #pre_model = '/path/to/your/pretrained/caffemodel'
    max_iter = 1000000
    #print pre_solver
    solver = caffe.get_solver(solver_file)
    
    #you can retrain the model base on the pretrained model
    if pre_solver!='':
      solver.restore(pre_solver)
    elif pre_model !='':
      solver.net.copy_from(pre_solver)
    # train the model according to the solver file
    solver.solve()
    
    #you can train the model step by step
    '''
    net = solver.net
    for i in range(0, max_iter):
        solver.step(1)
    '''
