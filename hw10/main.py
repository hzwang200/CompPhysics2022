import time
from scipy import *
from mweight import *
from numpy import linalg
from numpy import random
import numpy as np
from numba import jit
import matplotlib.pylab as plt

class params:
    def __init__(self):
        self.kF = 1.             # typical momentum
        self.cutoff = 3*self.kF  # integration cutoff
        self.dkF = 0.1*self.kF   # the size of a step
        #
        self.Nitt = 2000000   # total number of MC steps
        self.Ncout = 50000    # how often to print
        self.Nwarm = 1000     # warmup steps
        self.tmeassure = 10   # how often to meassure
        #
        self.Nbin = 129       # how many bins for saving the histogram
        self.V0norm = 2e-2    # starting V0
        self.dexp = 6         # parameter for fm at the first iteration, we will use 
        
        self.recomputew = 2e4/self.tmeassure # how often to check if V0 is correct
        self.per_recompute = 7 # how often to recompute fm auxiliary measuring function
        
@jit(nopython=True)
def TrialStep0(qx, momentum):
    tiQ = int( random.rand()*len(qx) )               # trial iQ for qx[iQ]
    Ka_new = qx[tiQ]                                 # |r_0| length of the vector
    th, phi = np.pi*random.rand(), 2*np.pi*random.rand()   # spherical angles for vector q in spherical coordinates
    sin_th = np.sin(th)                                 # trial step probability is proportional to sin(theta) when using sperical coodinates
    Q_sin_th = Ka_new * sin_th
    K_new = np.array([Q_sin_th*np.cos(phi), Q_sin_th*np.sin(phi), Ka_new*np.cos(th)]) # new 3D vector r_0
    q2_sin2_old = sum(momentum[0,:2]**2)    # x^2+y^2 = r_old^2 sin^2(theta_old)
    q2_old = q2_sin2_old + momentum[0,2]**2 # x^2+y^2+z^2 = r_old^2 
    trial_ratio = 1.
    if q2_old != 0:                               # make sure we do not get nan
        sin_th_old = np.sqrt(q2_sin2_old/q2_old)     # sin(theta_old)
        if sin_th_old != 0:                       # make sure we do not get nan
            trial_ratio = sin_th/sin_th_old
    accept=True
    return (K_new, Ka_new, trial_ratio, accept, tiQ)

@jit(nopython=True)
def TrialStep1(iloop,momentum,dkF,cutoff):
    dk = (2*random.rand(3)-1)*dkF    # change of k in cartesian coordinates of size p.dkF
    K_new = momentum[iloop,:] + dk     # K_new = K_old + dK
    Ka_new = linalg.norm(K_new)        # norm of the new vector
    trial_ratio = 1.                   # trial step probability is just unity
    accept = Ka_new <= cutoff        # we might reject the step if the point is too far from the origin
    return (K_new, Ka_new, trial_ratio, accept)

@jit(nopython=True)
def Give_new_X(momentum, K_new, iloop):
    tmomentum = np.copy(momentum)
    tmomentum[iloop,:] = K_new  # this is trial configuration X_{new}=momentum
    return tmomentum

@jit(nopython=True)
def ferm(x):
    if x>700:
        return 0.
    else: 
        return 1./(np.exp(x)+1.)
    
@jit(nopython=True)
def Linhard2_inside(momentum, Omega, kF, T, broad, nrm):
    q = momentum[0,:] # Notice that q is the first component of momenta
    k = momentum[1,:]
    e_k_q = linalg.norm(k-q)**2 - kF*kF
    e_k = linalg.norm(k)**2 - kF*kF
    dfermi = (ferm(e_k_q/T)-ferm(e_k/T))
    return -2*nrm*dfermi/(Omega-e_k_q+e_k+broad*1j)

class Linhard2:
    def __init__(self, Omega, kF, T, broad):
        self.Omega = Omega
        self.kF = kF
        self.T = T
        self.broad = broad
        self.nrm = 1/(2*np.pi)**3
        self.Ndim = 2  # Notice we need to add this for metropolis routine
    def __call__(self, momentum):
        return Linhard2_inside(momentum, self.Omega, self.kF, self.T, self.broad, self.nrm)
    
def IntegrateByMetropolis4(func, qx, p):
    """ Integration by Metropolis:
          func(momentum)   -- function to integrate
          qx               -- mesh given by a user
          p                -- other parameters
        Output:
          Pval(qx)
    """
    tm1 = time.time()
    random.seed(0)         # make sure that we always get the same sequence of steps
    Pval = np.zeros((len(qx),len(func.Omega)),dtype=complex)  # CHANGE: Final results V_physical is stored in Pval
    Pnorm = 0.0            # V_alternative is stored in Pnorm
    Pval_sum = 0.0         # this is widetilde{V_physical}
    Pnorm_sum = 0.0        # this is widetilde{V_alternative}
    V0norm = p.V0norm      # this is V0
    dk_hist = 1.0          # we are creating histogram by adding each configuration with weight 1.
    Ndim = func.Ndim       # dimensions of the problem
    inc_recompute = (p.per_recompute+0.52)/p.per_recompute # How often to self-consistently recompute
    # the wight functions g_i and h_{ij}.
    
    momentum = np.zeros((Ndim,3)) # contains all variables (r1,r2,r3,....r_Ndim)
    # We call them momentum here, but could be real space vectors or momentum space vectors.
    iQ = int(len(qx)*random.rand()) # which bin do we currently visit for r0, iQ is current r0=qx[iQ]
    momentum[1:,:] = random.random((Ndim-1,3)) * p.kF / np.sqrt(3.) # Initial guess for r1,r2,....r_N is random
    momentum[0,:] = [0,0,qx[iQ]]  # initial configuration for r_0 has to be consistent with iQ, and will be in z-direction

    # This is fm function, which is defined in mweight.py module
    mweight = meassureWeight(p.dexp, p.cutoff, p.kF, p.Nbin, Ndim) # measuring function fm in alternative space
    # fQ on the current configuration. Has two components (f(X), V0*f_m(X))
    fQ = func(momentum), V0norm * mweight( momentum )  # fQ=(f(X), V0*f_m(X))
    #print('starting with f=', fQ, '\nstarting momenta=', momentum)

    t_sim, t_mes, t_prn, t_rec = 0.,0.,0.,0.
    Nmeassure = 0  # How many measurements we had?
    Nall_q, Nall_k, Nall_w, Nacc_q, Nacc_k = 0, 0, 0, 0, 0
    c_recompute = 0 # when to recompute the auxiliary function?
    for itt in range(p.Nitt):   # long loop
        t0 = time.time()
        iloop = int( Ndim * random.rand() )   # which variable to change, iloop=0 changes external r_0
        accept = False
        if (iloop == 0):                      # changing external variable : r_0==Q
            Nall_q += 1                                      # how many steps changig external variable
            (K_new, Ka_new, trial_ratio, accept, tiQ) = TrialStep0(qx, momentum)
        else:   # changing momentum ik>0
            Nall_k += 1                        # how many steps of this type
            (K_new, Ka_new, trial_ratio, accept) = TrialStep1(iloop,momentum,p.dkF,p.cutoff)
        if (accept): # trial step successful. We did not yet accept, just the trial step.
            tmomentum = Give_new_X(momentum, K_new, iloop)
            fQ_new = func(tmomentum), V0norm * mweight(tmomentum) # f_new
            # Notice that we take |f_new(X)+V0*fm_new(X)|/|f_old(X)+V0*fm_old(X)| * trial_ratio
            ratio = (abs(fQ_new[0][0])+fQ_new[1])/(abs(fQ[0][0])+fQ[1]) * trial_ratio # !!CHANGE!!
            accept = abs(ratio) > 1-random.rand() # Metropolis
            if accept: # the step succeeded
                momentum[iloop] = K_new
                fQ = fQ_new
                if iloop==0:
                        Nacc_q += 1  # how many accepted steps of this type
                        iQ = tiQ     # the new external variable index
                else:
                        Nacc_k += 1  # how many accepted steps of this type
        t1 = time.time()
        t_sim += t1-t0
        if (itt >= p.Nwarm and itt % p.tmeassure==0): # below is measuring every p.tmeassure steps
            Nmeassure += 1   # new meassurements
            W = abs(fQ[0][0])+fQ[1]          # !!CHANGE!! this is the weight we are using
            f0, f1 = fQ[0]/W, fQ[1]/W        # the two measuring quantities
            Pval[iQ,:]+= f0                  # CHANGE: V_physical : integral up to a constant
            Pnorm     += f1                  # V_alternative : the normalization for the integral
            Pnorm_sum += f1                  # widetilde{V}_alternative, accumulated over all steps
            Wphs  = abs(f0[0])               # !!CHANGE!! widetilde{V}_{physical}, accumulated over all steps
            Pval_sum  += Wphs
            # doing histogram of the simulation in terms of V_physical only.
            # While the probability for a configuration is proportional to f(X)+V0*fm(X), the histogram for
            # constructing g_i and h_{ij} is obtained from f(X) only. 
            mweight.Add_to_K_histogram(dk_hist*Wphs, momentum, p.cutoff, p.cutoff)
            if itt>10000 and itt % (p.recomputew*p.tmeassure) == 0 :
                # Now we want to check if we should recompute g_i and h_{ij}
                # P_v_P is V_physical/V_alternative*0.1
                P_v_P = Pval_sum/Pnorm_sum * 0.1 
                # We expect V_physical/V_alternative*0.1=P_v_P to be of the order of 1.
                # We do not want to change V0 too much, only if P_V_P falls utside the
                # range [0.25,4], we should correct V0.
                change_V0 = 0
                if P_v_P < 0.25 and itt < 0.2*p.Nitt:  # But P_v_P above 0.25 is fine
                    change_V0 = -1  # V0 should be reduced
                    V0norm    /= 2  # V0 is reduced by factor 2
                    Pnorm     /= 2  # V_alternative is proportional to V0, hence needs to be reduced too. 
                    Pnorm_sum /= 2  # widetilde{V}_alternative also needs to be reduced
                if P_v_P > 4.0 and itt < 0.2*p.Nitt: # and P_v_P below 4 is also fine
                    change_V0 = 1   # V0 should be increased 
                    V0norm    *= 2  # actually increasing V0
                    Pnorm     *= 2
                    Pnorm_sum *= 2
                if change_V0:       # V0 was changed. Report that. 
                    schange = ["V0 reduced to ", "V0 increased to"]
                    print('%9.2fM P_v_P=%10.6f' % (itt/1e6, P_v_P), schange[int( (change_V0+1)/2 )], V0norm )
                    # Here we decied to drop all prior measurements if V0 is changed.
                    # We could keep them, but the convergence can be better when we drop them.
                    Pval = np.zeros(shape(Pval), dtype=complex)  # !!CHANGE!!
                    Pnorm = 0
                    Nmeasure = 0
                # Next we should check if g_i and h_ij need to be recomputed.
                # This should not be done too often, and only in the first half of the sampling.
                if (c_recompute==0 and itt<0.7*p.Nitt):
                    t5 = time.time()
                    # At the beginning we recompute quite often, later not so often anymore
                    # as the per_recompute is increasing...
                    p.per_recompute = int(p.per_recompute*inc_recompute+0.5)
                    # We normalized f_m, hence all previous accumulated values are now of the order
                    # of 1/norm. We also normalize the new additions to histogram with similar value, 
                    # but 5-times larger than before.
                    dk_hist *= 5*mweight.Normalize_K_histogram()
                    if dk_hist < 1e-8: # Once dk becomes too small, just start accumulating with weight 1.
                        dk_hist = 1.0
                    mweight.Recompute()# Here we actually recompute g_i and h_{ij}.
                    fQ = func(momentum), V0norm * mweight( momentum ) # And now we must recompute V0*f_m, because f_m has changed!            
                    #print(shape(fQ[0]),fQ[1])
                    t6 = time.time()
                    print('%9.2fM recomputing f_m=%10.6f' % (itt/1e6, fQ[1]))
                    t_rec += t6-t5
                c_recompute += 1
                if c_recompute>=p.per_recompute : c_recompute = 0 # counting when we will recompute next.
        t2 = time.time()
        t_mes += t2-t1
        if (itt+1)% p.Ncout == 0 : # This is just printing information
            P_v_P = Pval_sum/Pnorm_sum * 0.1 # what is curent P_v_P
            Qa = qx[iQ]                      # current r0
            ka = linalg.norm(momentum[1,:])  # current r1
            ratio = (abs(fQ_new[0][0])+fQ_new[1])/(abs(fQ[0][0])+fQ[1]) # CHANGE. current ratio
            print( '%9.2fM Q=%5.3f k=%5.3f fQ_new=%8.3g,%8.3g fQ_old=%8.3g,%8.3g P_v_P=%10.6f' % (itt/1e6, Qa, ka, abs(fQ_new[0][0]), fQ_new[1], abs(fQ[0][0]), fQ[1], P_v_P) ) # CHANGE
        t3 = time.time()
        t_prn += t3-t2
        
    Pval *= len(qx) * V0norm / Pnorm  # Finally, the integral is I = V0 *V_physical/V_alternative
    # This would be true if we are returning one single value. But we are sampling len(qx) values
    # And we jump between qx[i] uniformly, hence each value should be normalized with len(qx).
    tp1 = time.time()
    print('Total acceptance rate=', (Nacc_k+Nacc_q)/(p.Nitt+0.0), 'k-acceptance=', Nacc_k/(Nall_k+0.0), 'q-acceptance=', Nacc_q/(Nall_q+0.0))
    print('k-trials=', Nall_k/(p.Nitt+0.0), 'q-trial=', Nall_q/(p.Nitt+0.0) )
    print('t_simulate=%6.2f t_meassure=%6.2f t_recompute=%6.2f t_print=%6.2f t_total=%6.2f' % (t_sim, t_mes, t_rec, t_prn, tp1-tm1))
    return (Pval,mweight)

rs = 2.
kF = pow( 9*np.pi/4., 1./3.) / rs   # given in the homework, this is kF of the uniform electron gas
nF = kF/(2*np.pi*np.pi)                # this is density of states at the fermi level in uniform electron gas
T = 0.02*kF**2                   # temperature, as given in the homework
broad = 0.002*kF**2              # broadening, as given in the homework
cutoff = 3*kF                    # cutoff for the integration, as given in the homework

Omega = linspace(0,kF**2,100)
qx = linspace( 0.1*kF, 0.4*kF, 4)

lh2 = Linhard2(Omega, kF, T, broad)
  
p = params()
p.Nitt = 5000000 

(Pval,mweight) = IntegrateByMetropolis4(lh2, qx, p)

# plot
fig = plt.figure(figsize = (4, 4), dpi = 300)
ax = fig.add_subplot(1, 1, 1)
for iq in range(shape(Pval)[0]):
    ax.plot(Omega, Pval[iq,:].real/nF)
    ax.plot(Omega, Pval[iq,:].imag/nF)
plt.tight_layout()
plt.show()