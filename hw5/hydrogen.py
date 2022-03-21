import numpy as np
from scipy import *
from scipy import integrate as integ
from scipy import optimize as opt
import matplotlib.pyplot as plt

# part a
# u''(r) = (l * (l + 1) / r**2 - 2 * Z / r - En) * u(r), Z = 1
# v(r) = u'(r)
# v'(r) = (l * (l + 1) / r**2 - 2 / r - En) * u(r)
def Schroe(y, r, l, En): 
    u, v = y
    dydr = [v, (l * (l + 1) / r**2 - 2 / r - En) * u]
    return dydr

R = np.linspace(1e-10, 20, 500) 
l = 0 
E0 = -1. 

ur = integ.odeint(Schroe, [0., 1.], R, args = (l, E0)) 

# plot u(r), v(r)
fig = plt.figure(figsize = (3, 3), dpi = 300)
ax = fig.add_subplot(1, 1, 1)
ax.plot(R, ur[:, 0], 'C0', label = 'u(r)')
ax.plot(R, ur[:, 1], 'C1', label = r'$\frac{du(r)}{dr}$')
ax.set_xticks([0, 5, 10, 15, 20])
ax.legend(frameon = False)
plt.show()
# the integration is unstable when R is large

# part b
# do the integration from large R
R = np.linspace(1e-10, 100, 5000) 
Rb = R[::-1]

urb = integ.odeint(Schroe, [0., -1e-5], Rb, args = (l, E0))
ur = urb[:, 0][::-1]

# normalization
norm = integ.simps(ur**2, x = R)
ur *= 1./np.sqrt(norm)

# plot u(r), and zoom in plot
fig = plt.figure(figsize = (6, 3), dpi = 300)
ax = fig.add_subplot(1, 2, 1)
ax.plot(R, ur, 'C0', label = 'u(r)')
ax.legend(frameon = False)

ax = fig.add_subplot(1, 2, 2)
ax.set_xlim(0, 0.01)
ax.set_ylim(0, 0.01)
ax.set_xticks([0, 0.01])
ax.set_yticks([0, 0.01])
ax.plot(R, ur, 'C0', label = 'u(r)')
ax.legend(frameon = False)
plt.show()

def sol_Schroe(En, l, R): 
    Rb = R[::-1]
    v0 = -1e-5
    urb = integ.odeint(Schroe, [0., v0], Rb, args = (l, En))
    ur = urb[:, 0][::-1]
    norm = integ.simps(ur**2, x = R)
    ur *= 1./np.sqrt(norm)
    return ur

l = 1
En = -1./(2**2)

Ri = np.linspace(1e-6, 20, 500)
ui = sol_Schroe(En, l, Ri)

# plot ui(r)
fig = plt.figure(figsize = (3, 3), dpi = 300)
ax = fig.add_subplot(1, 1, 1)
ax.plot(Ri, ui, 'o-', label = 'u(r)')
ax.legend(frameon = False)
plt.show()

R = np.logspace(-5, 2., 500)
ur = sol_Schroe(En, l, R)

# plot u(r)
fig = plt.figure(figsize = (3, 3), dpi = 300)
ax = fig.add_subplot(1, 1, 1)
ax.plot(R, ur, 'o-', label = 'u(r)')
ax.set_xlim([0, 20])
ax.legend(frameon = False)
plt.show()

def shoot(En, R, l):
    Rb = R[::-1]
    v0 = -1e-5
    ub = integ.odeint(Schroe, [0.0, v0], Rb, args = (l,En))
    ur = ub[:,0][::-1]
    norm = integ.simps(ur**2, x = R)
    ur *= 1./np.sqrt(norm)
    
    ur = ur/R**l
    
    f0 = ur[0]
    f1 = ur[1]
    f_at_0 = f0 + (0. - R[0]) * (f1 - f0) / (R[1] - R[0])
    return f_at_0

R = np.logspace(-5, 2.2, 500)
print(shoot(-1./2**2, R, 1))

# part c
def boundstate(R, l, nmax, E_search):
    n = 0
    E_bnd = []
    u0 = shoot(E_search[0], R, l)
    for i in range(1, len(E_search)):
        u1 = shoot(E_search[i], R, l)
        if u0 * u1 < 0:
            E_bound = opt.brentq(shoot, E_search[i - 1], E_search[i], xtol = 1e-16, args = (R, l))
            E_bnd.append((l, E_bound))
            if len(E_bnd) > nmax:
                break
            n += 1
            print('Found bound state at E = %14.9f E_exact = %14.9f l = %d' % (E_bound, -1./(n+l)**2, l))
        u0 = u1
    return E_bnd

E_search = -1.2/np.arange(1, 20, 0.2)**2

R = np.logspace(-6, 2.2, 500)

nmax = 7
bnd = []
for l in range(nmax - 1):
    bnd += boundstate(R, l, nmax - l, E_search)

def take3rd(elem):
    return elem[0] + 2e3 * elem[1]

bnd.sort(key = take3rd)
# print(bnd)

# part d
Z = 28
N = 0
rho = np.zeros(len(R))

fig = plt.figure(figsize = (3, 3), dpi = 300)
ax = fig.add_subplot(1, 1, 1)

for (l,En) in bnd:
    ur = sol_Schroe(En, l, R)
    dN = 2 * (2 * l + 1)
    if N + dN <= Z:
        ferm = 1.
    else:
        ferm = (Z - N) / float(dN)
    drho = ur**2 * ferm * dN/(4*np.pi*R**2)
    rho += drho
    N += dN
    print('adding state (%2d, %14.9f) with fermi = %4.2f and current N = %5.1f' % (l, En, ferm, N))
    ax.plot(R, drho*(4*np.pi*R**2))
    if N >= Z: 
        break
ax.set_xlim([0, 25])
plt.show()

fig = plt.figure(figsize = (3, 3), dpi = 300)
ax = fig.add_subplot(1, 1, 1)
ax.plot(R, rho*(4*np.pi*R**2), label = 'charge density')
ax.set_xlim([0, 25])
plt.show()
