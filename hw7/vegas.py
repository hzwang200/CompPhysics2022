from numba import jit
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import det_curve

class Cumulants:
    def __init__(self):
        self.sum = 0.
        self.sqsum = 0.
        self.avg = 0.
        self.err = 0.
        self.chisq = 0.
        self.weightsum = 0.
        self.avgsum = 0.
        self.avg2sum = 0.

@jit(nopython = True)
def RefineGrid__(imp, self_g):
    (ndim, nbins) = np.shape(imp)
    gnew = np.zeros((ndim, nbins + 1))
    for idim in range(ndim):
        avgperbin = np.sum(imp[idim, :]) / nbins
        newgrid = np.zeros(nbins)
        cur = 0.
        newcur = 0.
        thisbin = 0.
        ibin = -1
        for newbin in range(nbins - 1):
            while (thisbin < avgperbin):
                ibin += 1
                thisbin += imp[idim, ibin]
                prev = cur
                cur = self_g[idim, ibin]
            thisbin -= avgperbin
            delta = (cur - prev) * thisbin
            newgrid[newbin] = cur - delta / imp[idim, ibin]

        newgrid[nbins - 1] = 1.
        gnew[idim, :nbins] = newgrid
    return gnew

class Grid:
    def __init__(self, ndim, nbins):
        self.g = np.zeros((ndim, nbins + 1))
        self.ndim = ndim
        self.nbins = nbins
        for idim in range(ndim):
            self.g[idim, :nbins] = np.arange(1, nbins + 1) / float(nbins)
    
    def RefineGrid(self, imp):
        self.g = RefineGrid__(imp, self.g)

def smfun(x):
    if (x > 0):
        return ((x - 1.) / np.log(x))**(1.5)
    else:
        return 0. 
vsmfun = np.vectorize(smfun)

def Smoothen(fxbin):
    (ndim, nbins) = np.shape(fxbin)
    final = np.zeros(np.shape(fxbin))
    for idim in range(ndim):
        fxb = np.copy(fxbin[idim, :])
        fxb[:nbins - 1] += fxbin[idim, 1:nbins]
        fxb[1:nbins] += fxbin[idim, :nbins - 1]
        fxb[1:nbins - 1] *= 1/3.
        fxb[0] *= 1/2.
        fxb[nbins - 1] *= 1/2.
        norm = np.sum(fxb)
        if(norm == 0):
            print('ERROR can not refine the grid with zero grid function')
            return
        fxb *= 1./norm
        final[idim, :] = vsmfun(fxb)
    return final

@jit(nopython = True)
def SetFxbin(fxbin, bins, wfun):
    (n, ndim) = bins.shape
    for dim in range(ndim):
        for i in range(n):
            fxbin[dim, bins[i, dim]] += abs(wfun[i])

def Vegas_step4(integrant, ab, maxeval, nstart, nincrease, grid, cum):
    ndim, nbins = grid.ndim, grid.nbins
    unit_dim = (ab[1] - ab[0])**ndim
    nbatch = 1000
    neval = 0
    print ("""Vegas parameters:
       ndim = """+str(ndim)+"""
       limits = """+str(ab)+"""
       maxeval = """+str(maxeval)+"""
       nstart = """+str(nstart)+"""
       nincrease = """+str(nincrease)+"""
       nbins = """+str(nbins)+"""
       nbaths = """+str(nbatch)+"\n")
    
    bins = np.zeros((nbatch, ndim), dtype = int)

    all_nsamples = nstart
    for iter in range(1000):
        wgh = np.zeros(nbatch)
        fxbin = np.zeros((ndim, nbins))
        for nsamples in range(all_nsamples, 0, -nbatch):
            n = min(nbatch, nsamples)
            xr = np.random.random((n, ndim))
            pos = xr * nbins
            bins = np.array(pos, dtype = int)
            wgh = np.ones(nbatch) / all_nsamples
            for dim in range(ndim):
                gi = grid.g[dim, bins[:, dim]]
                gm = grid.g[dim, bins[:, dim] - 1]
                diff = gi - gm
                gx = gm + (pos[:, dim] - bins[:, dim]) * diff
                xr[:, dim] = gx * (ab[1] - ab[0]) + ab[0]
                wgh *= diff*nbins

            fx = integrant(xr)
            neval += n

            wfun = wgh * fx
            cum.sum += np.sum(wfun)
            wfun *= np.conj(wfun)
            cum.sqsum += np.sum(wfun).real
            SetFxbin(fxbin, bins, wfun)
    
        w1 = cum.sqsum * all_nsamples - abs(cum.sum)**2
        w = (all_nsamples - 1) / w1
        cum.weightsum += w
        cum.avgsum += w * cum.sum

        cum.avg = cum.avgsum / cum.weightsum
        cum.err = np.sqrt(1 / cum.weightsum)

        chisq = 0
        if iter > 0:
            cum.chisq += abs(cum.sum - cum.avg)**2 * w
            chisq = cum.chisq / iter

        print("Iteration {:3d}: I = {:10.8f} +- {:10.8f} chisq = {:10.8f} number of evaluations = {:7d} ".format(iter + 1, cum.avg * unit_dim, cum.err * unit_dim, chisq, neval))
        imp = Smoothen(fxbin)
        grid.RefineGrid(imp)

        cum.sum = 0
        cum.sqsum = 0
        all_nsamples += nincrease
        if (neval >= maxeval):
            break
    
    cum.chisq *= 1. / iter
    cum.avg *= (ab[1] - ab[0])**ndim
    cum.err *= (ab[1] - ab[0])**ndim

def my_integrant2(x):
    return 1. / (1. - np.cos(x[:, 0]) * np.cos(x[:, 1]) * np.cos(x[:, 2])) / np.pi**3

ndim = 3
maxeval = 2000000
exact = 1.3932 * (2**3)

cum = Cumulants()

nbins = 128
nstart = 100000
nincrease = 5000

grids = Grid(ndim, nbins)

Vegas_step4(my_integrant2, [-np.pi, np.pi], maxeval, nstart, nincrease, grids, cum)

print(cum.avg, '+-', cum.err, 'exact = ', exact, 'real error = ', abs(cum.avg-exact) / exact)
 
# plot
fig = plt.figure(figsize = (3, 3), dpi = 300)
ax = fig.add_subplot(1, 1, 1)
ax.plot(grids.g[0, :nbins])
ax.plot(grids.g[1, :nbins])
ax.plot(grids.g[2, :nbins])
plt.show()   

@jit(nopython = True)
def ferm(x):
    if x > 700:
        return 0.
    else:
        return 1. / (np.exp(x) + 1.)

@jit(nopython = True)
def Linhard_inside(x, Omega, q, res, kF, T, broad, nrm):
    for i in range(x.shape[0]):
        k = x[i, 0:3]
        e_k_q = np.linalg.norm(k - q)**2 - kF * kF
        e_k = np.linalg.norm(k)**2 - kF * kF
        dfermi = (ferm(e_k_q / T) - ferm(e_k / T))
        res[i] = -2 * nrm * dfermi / (Omega - e_k_q + e_k + broad * 1j)
    return res

class Linhard:
    def __init__(self, Omega, q, kF, T, broad):
        self.Omega = Omega
        self.q = np.array([0, 0, q])
        self.kF = kF
        self.T = T
        self.broad = broad
        self.nrm = 1 / (2 * np.pi)**3
    
    def __call__(self, k):
        res = np.zeros(k.shape[0], dtype = complex)
        return Linhard_inside(k, self.Omega, self.q, res, self.kF, self.T, self.broad, self.nrm)
    
kF = 0.1
lh = Linhard(0.0, 0.1 * kF, kF, 0.02 * kF, 0.002 * kF)
x = np.random.random((4, 3))

rs = 2.
kF = pow(9 * np.pi / 4., 1./3.) / rs
nF = kF / (2 * np.pi * np.pi)
T = 0.02 * kF**2
broad = 0.002 * kF**2
cutoff = 3 * kF
Omega = 0. * kF**2
q = 0.1 * kF

ndim = 3
FCT = 1
maxeval = int(10000000 / FCT)
nbins = 128
nstart = int(200000 / FCT)
nincrease = int(100000 / FCT)

cum = Cumulants()
grids = Grid(ndim, nbins)
lh = Linhard(Omega, q, kF, T, broad)
Vegas_step4(lh, [-cutoff, cutoff], maxeval, nstart, nincrease, grids, cum)

print(cum.avg / nF, '+-', cum.err / nF)

# plot
fig = plt.figure(figsize = (3, 3), dpi = 300)
ax = fig.add_subplot(1, 1, 1)
ax.plot(grids.g[0, :nbins], 'o-')
ax.plot(grids.g[1, :nbins], 'o-')
ax.plot(grids.g[2, :nbins], 'o-')
plt.show()  

@jit(nopython = True)
def Linhard2_inside(x, Omega, q, res, kF, T, broad, nrm):
    for i in range(x.shape[0]):
        k = x[i, 0:3]
        e_k_q = np.linalg.norm(k - q)**2 - kF * kF
        e_k = np.linalg.norm(k)**2 - kF * kF
        dfermi = ferm(e_k_q / T) - ferm(e_k / T)
        res[:, i] = -2 * nrm * dfermi / (Omega - e_k_q + e_k + broad * 1j)
    return res

class Linhard2:
    def __init__(self, Omega, q, kF, T, broad):
        self.Omega = Omega
        self.q = np.array([0, 0, q])
        self.kF = kF
        self.T = T
        self.broad = broad
        self.nrm = 1 / (2 * np.pi)**3

    def __call__(self, x):
        res = np.zeros((len(self.Omega), x.shape[0]), dtype = complex)
        return Linhard2_inside(x, self.Omega, self.q, res, self.kF, self.T, self.broad, self.nrm)

Omega = np.linspace(0, kF**2, 10)
lh2 = Linhard2(Omega, q, kF, T, broad)
x = np.random.random((4, 3))
y = lh2(x)

class CumulantsW:
    def __init__(self, Om):
        nOm = len(Om)
        self.sum = np.zeros(nOm, dtype = complex)
        self.sqsum = np.zeros(nOm)
        self.avg = np.zeros(nOm, dtype = complex)
        self.err = np.zeros(nOm)
        self.chisq = np.zeros(nOm)
        self.weightsum = np.zeros(nOm)
        self.avgsum = np.zeros(nOm, dtype = complex)
        self.avg2sum = np.zeros(nOm)

def Vegas_step5(integrant, ab, maxeval, nstart, nincrease, grid, cum):
    ndim, nbins = grid.ndim, grid.nbins
    unit_dim = (ab[1] - ab[0])**ndim
    nbatch = 1000
    neval = 0
    print ("""Vegas parameters:
       ndim = """+str(ndim)+"""
       limits = """+str(ab)+"""
       maxeval = """+str(maxeval)+"""
       nstart = """+str(nstart)+"""
       nincrease = """+str(nincrease)+"""
       nbins = """+str(nbins)+"""
       nbaths = """+str(nbatch)+"\n")
    
    bins = np.zeros((nbatch, ndim), dtype = int)

    all_nsamples = nstart
    for iter in range(1000):
        wgh = np.zeros(nbatch)
        fxbin = np.zeros((ndim, nbins))
        for nsamples in range(all_nsamples, 0, -nbatch):
            n = min(nbatch, nsamples)
            xr = np.random.random((n, ndim))
            pos = xr * nbins
            bins = np.array(pos, dtype = int)
            wgh = np.ones(nbatch) / all_nsamples
            for dim in range(ndim):
                gi = grid.g[dim, bins[:, dim]]
                gm = grid.g[dim, bins[:, dim] - 1]
                diff = gi - gm
                gx = gm + (pos[:, dim] - bins[:, dim]) * diff
                xr[:, dim] = gx * (ab[1] - ab[0]) + ab[0]
                wgh *= diff*nbins

            fx = integrant(xr)
            neval += n

            wfun = fx * wgh
            cum.sum += np.sum(wfun, axis = 1)
            wfun2 = abs(wfun * np.conj(wfun)) * all_nsamples
            cum.sqsum += np.sum(wfun2, axis = 1).real
            SetFxbin(fxbin, bins, wfun2[0, :])
    
        w1 = cum.sqsum - abs(cum.sum)**2
        w = (all_nsamples - 1) / w1
        cum.weightsum += w
        cum.avgsum += w * cum.sum
        cum.avg2sum += w * abs(cum.sum)**2

        cum.avg = cum.avgsum / cum.weightsum
        cum.err = np.sqrt(1 / cum.weightsum)

        chisq = 0
        if iter > 0:
            cum.chisq += abs(cum.sum - cum.avg)**2 * w
            chisq = cum.chisq[0] / iter

        print("Iteration {:3d}: I = {:10.8f} +- {:10.8f} chisq = {:10.8f} number of evaluations = {:7d} ".format(iter + 1, cum.avg[0] * unit_dim, cum.err[0] * unit_dim, chisq, neval))
        imp = Smoothen(fxbin)
        grid.RefineGrid(imp)

        cum.sum[:] = 0
        cum.sqsum[:] = 0
        all_nsamples += nincrease
        if (neval >= maxeval):
            break
    
    cum.chisq *= 1. / iter
    cum.avg *= unit_dim
    cum.err *= unit_dim

Omega = np.linspace(0, 0.5 * kF**2, 100)
lh2 = Linhard2(Omega, q, kF, T, broad)
cum = CumulantsW(Omega)

ndim = 3
grids = Grid(ndim, nbins)

Vegas_step5(lh2, [-cutoff, cutoff], maxeval, nstart, nincrease, grids, cum)

# plot
fig = plt.figure(figsize = (3, 3), dpi = 300)
ax = fig.add_subplot(1, 1, 1)
ax.errorbar(Omega, cum.avg.real / nF, yerr = cum.err)
ax.errorbar(Omega, cum.avg.imag / nF, yerr = cum.err)
plt.show()  

print(cum.avg[0] / nF, '+-', cum.err[0] / nF)

# plot
fig = plt.figure(figsize = (3, 3), dpi = 300)
ax = fig.add_subplot(1, 1, 1)
ax.plot(grids.g[0, :nbins], 'o-')
ax.plot(grids.g[1, :nbins], 'o-')
ax.plot(grids.g[2, :nbins], 'o-')
plt.show() 