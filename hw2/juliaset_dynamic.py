#!/usr/bin/env python

from scipy import *
from pylab import *
from numpy import *
import time
from numba import jit

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

def ax_update(ax):
    ax.set_autoscale_on(False)
    xstart, ystart, xdelta, ydelta = ax.viewLim.bounds
    xend = xstart + xdelta
    yend = ystart + ydelta
    data = JuliasetNumba(width, height, zoom, cX, cY, moveX, moveY, maxIter)
    
    im = ax.images[-1]
    im.set_data(data)
    ax.figure.canvas.draw_idle()


if __name__ == "__main__":

    width = 1920
    height = 1080
    zoom = 1
    cX, cY = -0.7, 0.27015
    moveX, moveY = 0.0, 0.0
    maxIter = 255

    t0 = time.time()
    data = JuliasetNumba(width, height, zoom, cX, cY, moveX, moveY, maxIter)
    t1 = time.time()
    print('Python: ', t1-t0)

    #imshow(1./data)
    fig,ax=subplots(1,1)
    ax.imshow(data, aspect='equal', origin='lower')
    
    ax.callbacks.connect('xlim_changed', ax_update)
    ax.callbacks.connect('ylim_changed', ax_update)
    
    show()

