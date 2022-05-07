from dis import dis
import numpy as np
from numpy import random
from numpy import linalg
from numba import jit
from pylab import *

def Distance(R1, R2):
    return linalg.norm(R1 - R2)

def TotalDistance(city, R):
    dist = 0
    for i in range(len(city) - 1):
        dist += Distance(R[city[i]], R[city[i + 1]])
    dist += Distance(R[city[-1]], R[city[0]])
    return dist

def Plot(city, R, dist):
    Pt = [R[city[i]] for i in range(len(city))]
    Pt += [R[city[0]]]
    Pt = np.array(Pt)
    title('Total distance = '+str(dist))
    plot(Pt[:, 0], Pt[:, 1], 'o-')
    show()

@jit(nopython = True)
def FindASegment(R):
    nct = len(R) # number of cities
    while True:
        # Two cities n[0] and n[1] chosen at random
        n0 = int(nct * rand())
        n1 = int((nct - 1) * rand())
        if n1 >= n0 : n1 += 1
        if n1 < n0 : (n0, n1) = (n1, n0)
        nn = nct - (n1 - n0 + 1)  # the rest of the cities
        if nn >= 3 : break
    n2 = (n0 - 1) % nct
    n3 = (n1 + 1) % nct
    return (n0, n1, n2, n3)

def CostReverse(R, city, n0, n1, n2, n3):
    # cost for reverse move
    de = Distance(R[city[n2]], R[city[n1]]) + Distance(R[city[n0]], R[city[n3]])
    de -= Distance(R[city[n2]], R[city[n0]]) + Distance(R[city[n1]], R[city[n3]])
    return de

def Reverse(R, city, n0, n1, n2, n3):
    newcity = np.copy(city)
    for j in range(n1 - n0 + 1):
        newcity[n0 + j] = city[n1 - j]
    return newcity

@jit(nopython = True)
def FindTSegment(R):
    (n0, n1, n2, n3) = FindASegment(R)
    nct = len(R)
    nn = nct - (n1 - n0 + 1)  # number for the rest of the cities
    n4 = (n1 + 1 + int(rand()*(nn - 1)) ) % nct # city on the rest of the path
    n5 = (n4 + 1) % nct
    return (n0, n1, n2, n3, n4, n5)

def CostTranspose(R, city, n0, n1, n2, n3, n4, n5):
    de = -Distance(R[city[n1]], R[city[n3]])
    de -= Distance(R[city[n0]], R[city[n2]])
    de -= Distance(R[city[n4]], R[city[n5]])
    de += Distance(R[city[n0]], R[city[n4]])
    de += Distance(R[city[n1]], R[city[n5]])
    de += Distance(R[city[n2]], R[city[n3]])
    return de

def Transpose(R, city, n0, n1, n2, n3, n4, n5):
    nct = len(R)
    newcity = []
    # Segment in the range n0,...n1
    for j in range(n1 - n0 + 1):
        newcity.append(city[(j + n0) % nct])
    # is followed by segment n5...n2
    for j in range((n2 - n5) % nct + 1):
        newcity.append(city[(j + n5) % nct])
    # is followed by segement n3..n4
    for j in range((n4 - n3) % nct + 1):
        newcity.append(city[(j + n3) % nct])
    return newcity

def TravelingSalesman(city, R, maxSteps, maxAccepted, Tstart, fCool, maxTsteps, Preverse = 0.5):
    T = Tstart
    dist = TotalDistance(city, R)
    for t in range(maxTsteps):
        accepted = 0
        for i in range(maxSteps):
            if Preverse > rand():
                # Try reverse
                nn = FindASegment(R)
                de = CostReverse(R, city, *nn)
                if de < 0 or np.exp(-de / T) > rand():
                    accepted += 1
                    dist += de
                    city = Reverse(R, city, *nn)
            else: 
                # here we transpose
                nn = FindTSegment(R)
                de = CostTranspose(R, city, *nn)
                if de < 0 or np.exp(-de / T) > rand():
                    accepted += 1
                    dist += de
                    city = Transpose(R, city, *nn)
            if accepted > maxAccepted: 
                break    
        T *= fCool
        # Plot(city, R, dist)
        print("T = %10.5f, distance = %10.5f, acc.steps = %d" % (T, dist, accepted))
        if accepted == 0:
            break
    Plot(city, R, dist)
    return (city, dist) 

dist_tot = []
for i in range (0, 100, 1):
    
    ncity = 10 * (i + 1)
    maxSteps = 100 * ncity
    maxAccepted = 10 * ncity
    
    Tstart = 0.2
    fCool = 0.9
    maxTsteps = 100
    
    np.random.seed(0)
    
    R = np.random.random((ncity, 2))
    city = range(ncity)
    
    Plot(city, R, TotalDistance(city, R))
    
    nn = FindTSegment(R)
    de = CostTranspose(R, city, *nn)
    
    print(de)
    r1 = Transpose(R, city, *nn)
    print(r1)
    
    (ncity, dist) = TravelingSalesman(city, R, maxSteps, maxAccepted, Tstart, fCool, maxTsteps)
    dist_tot.append(dist)
    # print(dist_tot)

ncity_num = np.linspace(10, 1000, num = 100)
# plot d vs ncity
fig = figure(figsize = (4, 4), dpi = 300)
ax = fig.add_subplot(1, 1, 1)
ax.plot(ncity_num, dist_tot)
ax.set_xlabel('# of cities')
ax.set_ylabel('total distance')
tight_layout()
show()