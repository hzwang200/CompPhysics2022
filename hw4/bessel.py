from scipy import *
from numpy import *

def GetPolyInt(z, alpha, n):
    small = 1e-12
    how_small = 0.5
    x = -alpha/z
    Kn = zeros(n + 1, dtype=type(z))
    if n == 0:
        Kn[0] = (log(z + alpha) - log(z))/alpha
        return Kn
    if abs(x) < how_small: 
        # Taylor expansion
        x_n = 1.0
        KN = 0
        for m in range(1000):
            dK = 1.0/(n + m + 1)*(x_n/z)
            KN += dK
            x_n *= x
            if abs(dK) < small: break
        Kn [n] = KN
        # downward recursion
        for np1 in range(n, 0, -1):
            Kn[np1 - 1] = 1.0/(np1*z) + x * Kn[np1]
    else:
        # upward recursion
        Kn[0] = 1/alpha * log((z + alpha)/z)
        for i in range(n):
            Kn[i + 1] = 1/alpha * (1.0/(i + 1.) - z * Kn[i])
    return Kn

n = 10
z = 2. 
alpha = 200
print('alpha/z = ', alpha/z)
print(GetPolyInt(z, alpha, n))
alpha = 2 * 1e-4
print('alpha/z = ', alpha/z)
print(GetPolyInt(z, alpha, n))