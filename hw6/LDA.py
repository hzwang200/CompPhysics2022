from zlib import Z_DEFAULT_COMPRESSION
import numpy as np
from scipy import *
from scipy import integrate as integ
from scipy import optimize as opt
from scipy import interpolate
import matplotlib.pyplot as plt
from excor import ExchangeCorrelation

def Numerovc(f, x0, dx, dh):
    x = np.zeros(len(f))
    dh = float(dh)
    x[0] = x0
    x[1] = x0 + dh * dx

    h2 = dh * dh
    h12 = h2 / 12.
    
    w0 = x[0] * (1 - h12 * f[0])
    w1 = x[1] * (1 - h12 * f[1])
    xi = x[1]
    fi = f[1]
    for i in range(2, len(f)): 
        w2 = 2 * w1 - w0 + h2 * fi *xi
        fi = f[i]
        xi = w2 / (1 - h12 * fi)
        x[i] = xi
        w0 = w1
        w1 = w2
    return x

def NumerovU(U, x0, dx, dt):
    x = np.zeros(len(U))
    x[0] = x0
    x[1] = dx * dt + x0
    h2 = dt * dt
    h12 = h2 / 12.
    w0 = x[0] - h12 * U[0]
    w1 = x[1] - h12 * U[1]
    xi = x[1]
    Ui = U[1]
    for i in range(2, len(U)):
        w2 = 2 * w1 - w0 + h2 * Ui
        Ui = U[i]
        xi = w2 + h12 * Ui
        x[i] = xi
        w0 = w1
        w1 = w2
    return x

def fSchrod2(En, R, l, Uks):
    return l * (l + 1.) / R**2 + Uks / R - En

def ComputeSchrod(En, R, l, Uks):
    f = fSchrod2(En, R[::-1], l, Uks[::-1])
    ur = Numerovc(f, 0.0, -1e-7, -R[1] + R[0])[::-1]
    norm = integ.simps(ur**2, x = R)
    return ur * 1 / np.sqrt(abs(norm))

def Shoot(En, R, l, Uks):
    ur = ComputeSchrod(En, R, l, Uks)
    f0 = ur[0]
    f1 = ur[1]
    f_at_0 = f0 + (f1 - f0) * (0.0 - R[0]) / (R[1] - R[0])
    return f_at_0

def FindBoundStates(R, l, nmax, Esearch, Uks):
    n = 0
    Ebnd = []
    u0 = Shoot(Esearch[0], R, l, Uks)
    for i in range(1, len(Esearch)):
        u1 = Shoot(Esearch[i], R, l, Uks)
        if u0 * u1 < 0: 
            Ebound = opt.brentq(Shoot, Esearch[i-1], Esearch[i], xtol = 1e-16, args = (R, l, Uks))
            Ebnd.append((l, Ebound))
            if len(Ebnd) > nmax: break
            n += 1
            print('Found bound state at E = %14.9f E[Hartree] = %14.9f l = %d' % (Ebound, Ebound / 2, l))
        u0 = u1

    return Ebnd

def take3rd(elem): 
    return elem[0] + 2e3 * elem[1]

def ChargeDensity(bst, R, Zatom, Uks):
    rho = np.zeros(len(R))
    N = 0
    for i, (l, Ei) in enumerate(bst):
        dN = 2 * (2 * l + 1)
        if N + dN < Zatom: 
            ferm = 1 
        else: 
            ferm = (Zatom - N) / float(dN) 
        u = ComputeSchrod(Ei, R, l, Uks)
        drho = u**2 / (4 * np.pi * R**2) * dN * ferm
        rho += drho
        N += dN
        print('Adding state with l = ', l, 'and E = ', Ei / 2, ' Hartree with Z = ', N, 'with ferm = ', ferm)
        if N >= Zatom: 
            break
    return rho

def HartreeU(R, rho):
    ux = -8 * np.pi * R * rho
    dudx = 0.1
    U = NumerovU(ux, 0.0, dudx, R[1] - R[0])
    alpha2 = (2 * Zatom - U[-1]) / R[-1]
    U += alpha2 * R
    return U

def rs(rho):
    if rho < 1e-100:
        return 1e100
    return pow(3 / (4 * np.pi * rho), 1 / 3.)

R = np.linspace(1e-8, 50, 2**12 + 1) 

nmax = 2 
Zatom = 4

E0 = -1.2 * Zatom**2
Eshift = 0.5
Esearch = -np.logspace(-4, np.log10(-E0 + Eshift), 200)[::-1] + Eshift

exc = ExchangeCorrelation()
Uks = -2. * np.ones(len(R))

for itt in range(2):
    Bnd = []
    for l in range(nmax - 1): 
        Bnd += FindBoundStates(R, l, nmax - l, Esearch, Uks)
    Bnd.sort(key = take3rd)
    
    rho = ChargeDensity(Bnd, R, Zatom, Uks)
    U = HartreeU(R, rho)

    Vxc = [2 * exc.Vx(rs(rh)) + 2 * exc.Vc(rs(rh)) for rh in rho]

    Uks = U - 2 * Zatom + Vxc * R

    print('Total density has weight = ', integ.simps(rho * (4 * np.pi * R**2), x = R))

    fig = plt.figure(figsize = (3, 3), dpi = 300)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(R, U, label = 'U-hartree')
    ax.plot(R, Vxc, label = 'Vxc')
    ax.plot(R, Uks, label = 'Uks')
    ax.set_xlim([0, 50])
    ax.set_ylim([-8, 8])
    ax.grid()
    ax.legend()
    plt.show()
    
    fig = plt.figure(figsize = (3, 3), dpi = 300)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(R, rho * (4 * np.pi * R**2))
    ax.set_xlim([0, 20])
    plt.show()

# charge density mixing
R = np.linspace(1e-8, 50, 2**12 + 1)

nmax = 2
# for helium
Zatom = 4
mixr = 0.5

E0 = -1.2 * Zatom**2
Eshift = 0.5
Esearch = -np.logspace(-4, np.log10(-E0 + Eshift), 200)[::-1] + Eshift

exc = ExchangeCorrelation()
Uks = -2.*np.ones(len(R))

for itt in range(30):
    Bnd = []
    for l in range(nmax - 1):
        Bnd += FindBoundStates(R, l, nmax - l, Esearch, Uks)
    Bnd.sort(key = take3rd)

    rho_new = ChargeDensity(Bnd, R, Zatom, Uks)

    if itt > 0:
        rho = rho_new * mixr + (1 - mixr) * rho_old
    else:
        rho = rho_new
    rho_old = np.copy(rho_new)

    U = HartreeU(R, rho)

    Vxc = [2 * exc.Vx(rs(rh)) + 2 * exc.Vc(rs(rh)) for rh in rho]

    Uks = U - 2 * Zatom + Vxc * R

    print('Total density has weight = ', integ.simps(rho * (4 * np.pi * R**2), x = R))

fig = plt.figure(figsize = (3, 3), dpi = 300)
ax = fig.add_subplot(1, 1, 1)
ax.plot(R, U, label = 'U-hartree')
ax.plot(R, Vxc, label = 'Vxc')
ax.plot(R, Uks, label = 'Uks')
ax.set_xlim([0, 50])
ax.set_ylim([-8, 8])
ax.legend()
ax.grid()
plt.show()

fig = plt.figure(figsize = (3, 3), dpi = 300)
ax = fig.add_subplot(1, 1, 1)
ax.plot(R, rho * (4 * np.pi * R**2))
ax.set_xlim([0, 10])
plt.show()

def ChargeDensity(bst, R, Zatom, Uks):
    rho = np.zeros(len(R))
    N = 0
    Ebs = 0.
    for i, (l, Ei) in enumerate(bst):
        dN = 2 * (2 * l + 1)
        if N + dN < Zatom: 
            ferm = 1
        else: 
            ferm = (Zatom - N) / float(dN)
        u = ComputeSchrod(Ei, R, l, Uks)
        drho = u**2 / (4 * np.pi * R**2) * dN * ferm
        rho += drho
        N += dN
        Ebs += Ei * dN * ferm
        print('Adding state with l = ', l, 'and E = ', Ei / 2, ' Hartree with Z = ', N, 'with ferm = ', ferm)
        if N >= Zatom:
            break
    return (rho, Ebs)

R = np.linspace(1e-8, 10, 2**13 + 1)
Etol = 1e-7
nmax = 3
Zatom = 8
mixr = 0.5

E0 = -1.2 * Zatom**2
Eshift = 0.5
Esearch = -np.logspace(-4, np.log10(-E0 + Eshift), 200)[::-1] + Eshift

exc = ExchangeCorrelation()
Uks = -2.*np.ones(len(R))
Eold = 0

for itt in range(100):
    Bnd = []
    for l in range(nmax - 1):
        Bnd += FindBoundStates(R, l, nmax - l - 1, Esearch, Uks)
    Bnd.sort(key = take3rd)

    (rho_new, Ebs) = ChargeDensity(Bnd, R, Zatom, Uks)

    if itt > 0:
        rho = rho_new * mixr + (1 - mixr) * rho_old
    else:
        rho = rho_new
    rho_old = np.copy(rho_new)

    U = HartreeU(R, rho)
    Vxc = [2 * exc.Vx(rs(rh)) + 2 * exc.Vc(rs(rh)) for rh in rho]
    Uks = U - 2 * Zatom + Vxc * R

    ExcVxc = np.array([2 * exc.EcVc(rs(rh)) + 2 * exc.ExVx(rs(rh)) for rh in rho])
    pot = (ExcVxc * R**2 - 0.5 * U * R) * rho * 4 * np.pi
    Etot = integ.romb(pot, R[1] - R[0]) + Ebs

    print('Iteration', itt, 'Etot[Ry] = ', Etot, 'Etot[Hartree] = ', Etot / 2, 'Diff = ', abs(Etot - Eold))

    if itt > 0 and abs(Etot - Eold) < Etol:
        break
    Eold = Etot

    print('Total density has weight = ', integ.simps(rho * (4 * np.pi * R**2), x = R))

fig = plt.figure(figsize = (3, 3), dpi = 300)
ax = fig.add_subplot(1, 1, 1)
ax.plot(R, rho * (4 * np.pi * R**2), '--', label = 'rho')
ax.legend()
ax.set_xlim([0, 10])
plt.show()


