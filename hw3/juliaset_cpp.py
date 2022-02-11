#!/usr/bin/env python
from scipy import *
from pylab import *
from numpy import *
import time
import ijulia
from timeit import default_timer as timer
from numba import jit

def JuliasetPyth(width, height, zoom, cX, cY, moveX, moveY, maxIter):
    data = ones( (width, height))*maxIter
    for x in range(width):
        for y in range(height):
            zx = 1.5*(x - width/2)/(0.5*zoom*width) + moveX
            zy = 1.0*(y - height/2)/(0.5*zoom*height) + moveY
            i = maxIter
            while zx*zx + zy*zy < 4 and i > 1:
                tmp = zx*zx - zy*zy + cX
                zy,zx = 2.0*zx*zy + cY, tmp
                i -= 1
            data[x,y] = (i << 21) + (i << 10) + i*8
    return data

@jit(nopython=True)
def JuliasetNumba(width, height, zoom, cX, cY, moveX, moveY, maxIter):
    data = ones( (width, height))*maxIter
    for x in range(width):
        for y in range(height):
            zx = 1.5*(x - width/2)/(0.5*zoom*width) + moveX
            zy = 1.0*(y - height/2)/(0.5*zoom*height) + moveY
            i = maxIter
            while zx*zx + zy*zy < 4 and i > 1:
                tmp = zx*zx - zy*zy + cX
                zy,zx = 2.0*zx*zy + cY, tmp
                i -= 1
            data[x,y] = (i << 21) + (i << 10) + i*8
    return data

def JuliaPybind11(width, height, zoom, cX, cY, moveX, moveY, maxIter):
    data = ones( (width, height) );
    ijulia.julia(data, width, height, zoom, cX, cY, moveX, moveY, maxIter)
    return data


if __name__ == "__main__":

    width = 1920
    height = 1080
    zoom = 1
    cX, cY = -0.7, 0.27015
    moveX, moveY = 0.0, 0.0
    maxIter = 255

    t0 = time.time()
    t0_ = time.process_time()
    data = JuliaPybind11(width, height, zoom, cX, cY, moveX, moveY, maxIter)
    t1 = time.time()
    t1_ = time.process_time()
    print('pybind11: walltime: ', t1-t0, 'cputime: ', t1_-t0_)
    imshow(data)
    show()

    t0 = time.time()
    t0_ = time.process_time()
    data = JuliasetNumba(width, height, zoom, cX, cY, moveX, moveY, maxIter)
    t1 = time.time()
    t1_ = time.process_time()
    print('numba: walltime: ', t1-t0, 'cputime: ', t1_-t0_)
    imshow(1./data)
    show()

