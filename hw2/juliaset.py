#!/usr/bin/env python

from scipy import *
from pylab import *
from PIL import Image
import time

if __name__ == "__main__":

    width = 1920
    height = 1080
    zoom = 1

    t0 = time.time()
    
    bitmap = Image.new("RGB", (width, height), "white")
    
    pix = bitmap.load()
    
    cX, cY = -0.7, 0.27015
    moveX, moveY = 0.0, 0.0
    maxIter = 255
    
    for x in range(width):
        for y in range(height):
            zx = 1.5*(x - width/2)/(0.5*zoom*width) + moveX
            zy = 1.0*(y - height/2)/(0.5*zoom*height) + moveY
            i = maxIter
            while zx*zx + zy*zy < 4 and i > 1:
                tmp = zx*zx - zy*zy + cX
                zy,zx = 2.0*zx*zy + cY, tmp
                i -= 1

            pix[x,y] = (i << 21) + (i << 10) + i*8

    bitmap.show()

    print ('clock time: '+str( time.time()-t0) )
    
    
