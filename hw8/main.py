import numpy as np
from numpy import random
from numba import jit
import matplotlib.pyplot as plt

@jit(nopython = True)
def CEnergy(latt):
    N = np.shape(latt)[0]
    Ene = 0
    for i in range(len(latt)):
        for j in range(len(latt)):
            S = latt[i, j]
            WF = latt[(i+1)%N, j] + latt[i, (j+1)%N] + latt[(i-1)%N, j] + latt[i, (j-1)%N]
            Ene += -S * WF
    return Ene/2.

def RandomL(N):
    return np.array(np.sign(2*random.random((N, N))-1), dtype=int) 

def PrepareEnergies(N):
    Energies = (np.array(4*np.arange(-int(N*N/2), int(N*N/2)+1), dtype=int)).tolist()
    Energies.pop(1)  
    Energies.pop(-2) 
    Energies = np.array(Energies) 
    Emin, Emax = Energies[0],Energies[-1]
    indE = -np.ones(Emax+1-Emin, dtype=int) 
    for i, E in enumerate(Energies):
        indE[E-Emin] = i
    return (Energies, indE, Emin)

def WangLandau(Nitt, N, flatness):
    (Energies, indE, Emin) = PrepareEnergies(N)
    latt = RandomL(N)
    return RunWangLandau(Nitt, Energies, latt, indE)

@jit(nopython = True)
def RunWangLandau(Nitt, Energies, latt, indE):
    N   = len(latt)
    Ene = int(CEnergy(latt))
    Emin, Emax = Energies[0], Energies[-1]
    lngE = np.zeros(len(Energies))
    Hist = np.zeros(len(Energies))
    lnf = 1.0
    N2 = N*N
    for itt in range(Nitt):
        t = int(random.rand()*N2)
        (i, j) = (int(t/N), t%N)
        S = latt[i, j]
        WF = latt[(i+1)%N, j] + latt[i, (j+1)%N] + latt[(i-1)%N, j] + latt[i, (j-1)%N]
        Enew = Ene + int(2*S*WF)
        lgnew = lngE[indE[Enew-Emin]]
        lgold = lngE[indE[Ene-Emin]]
        P = 1.0
        if lgold-lgnew < 0: 
            P = np.exp(lgold-lgnew)
        if P > random.rand():
            latt[i, j] = -S
            Ene = Enew
        Hist[indE[Ene-Emin]] += 1
        lngE[indE[Ene-Emin]] += lnf
        
        if (itt+1) % 1000 == 0: 
            aH = sum(Hist)/N2 
            mH = min(Hist)
            if mH > aH*flatness: 
                Hist[:] = 0
                lnf /= 2.
                print(itt, 'histogram is flat', mH, aH, 'f = ', np.exp(lnf))
    return (lngE, Hist)

def RenG(lngE):
    if lngE[-1] < lngE[0]:
        lgC = np.log(4) - lngE[-1] - np.log(1 + np.exp(lngE[0] - lngE[-1]))
    else:
        lgC = np.log(4) - lngE[0] - np.log(1 + np.exp(lngE[-1] - lngE[0]))
    lngE += lgC
    return lngE

flatness = 0.9
N = 32
Nitt = int(1e9)

(Energies, indE, Emin) = PrepareEnergies(N)
latt = RandomL(N)

n = 5
SiE = np.zeros((n, len(Energies)))

for i in range(n):
    lngE, Hist = RunWangLandau(Nitt, Energies, latt, indE)
    SiE[i, :] = RenG(lngE)

Sa = np.sum(SiE, axis = 0)/n
S2a = np.sum(SiE**2, axis = 0)/n
sgm = np.sqrt((S2a - Sa**2)/n)

# plot
fig = plt.figure(figsize = (4, 4), dpi = 300)
ax = fig.add_subplot(1, 1, 1)

for i in range(n):    
    ax.plot(SiE[i] - Sa, label = 'S_{i = '+str(i)+'} - <S>')
ax.plot(sgm, label = 'sigma')
ax.legend(loc = 'best')
plt.tight_layout()
plt.show()

print([min(sgm), max(sgm), np.sum(sgm)/len(sgm)])

gExact = [2, 2048, 4096, 1057792, 4218880, 371621888, 2191790080, 100903637504, 768629792768, 22748079183872]

# plot
fig = plt.figure(figsize = (4, 4), dpi = 300)
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.log(gExact), label = r'$S_{exact}$')
ax.plot(Sa[:10], label = r'$S_{numeric}$')
ax.plot(np.log(gExact) - Sa[:10], label = 'diff')
ax.legend(loc = 'best')
plt.tight_layout()
plt.show()

# plot
fig = plt.figure(figsize = (4, 4), dpi = 300)
ax = fig.add_subplot(1, 1, 1)
ax.plot(abs(Sa[:10] - np.log(gExact)), label = '<S(E)> - S_{exact}')
ax.plot(sgm[:10], label = 'sigma')
ax.legend()
plt.tight_layout()
plt.show()

(Energies, indE, Emin) = PrepareEnergies(N)
def Thermod(T, lngE, Energies, N):
    Z = 0.
    Ev = 0.
    E2v = 0.
    for i, E in enumerate(Energies):
        w = np.exp(lngE[i] - lngE[0] - (E-Energies[0])/T)
        Z += w
        Ev += w*E
        E2v += w*E**2
    Ev *= 1./Z
    E2v *= 1./Z
    cv = (E2v-Ev**2)/T**2
    Entropy = np.log(Z) + lngE[0] - Energies[0]/T + Ev/T
    return (Ev/(N**2), cv/(N**2), Entropy/(N**2))

Te = np.linspace(0.5, 4., 300)

Thm = []
for T in Te:
    Thm.append(Thermod(T, Sa, Energies, N))
Thm = np.array(Thm)

# plot
fig = plt.figure(figsize = (4, 4), dpi = 300)
ax = fig.add_subplot(1, 1, 1)
ax.plot(Te, Thm[:, 0], label = 'E(T)')
ax.plot(Te, Thm[:, 1], label = 'cv(T)')
ax.plot(Te, Thm[:, 2], label = 'Entropy(T)')
ax.set_xlabel('T')
ax.legend(loc = 'best')
plt.tight_layout()
plt.show()