import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cs
import math

### constants ###
G = cs.gravitational_constant
hbar = cs.hbar
c = cs.speed_of_light
epsilon0 = cs.epsilon_0
kb = cs.Boltzmann
g = 2 # electronic g-factor
mub = 9.3*10**(-24) # Bohr magneton

### some experimental parameters ###
t = 0.25 # time of creation/annihilation of superpositions
tau = 0.5 # time that the superposition size is constant 
W = 1*10**(-6) # width of hte plate
p = 10**(-2)*1.602176634*10**(-19)*10**(-2) # assumed permanent dipole of the test mass
epsilon=5.7 # dielectric constant test mass
sdensity=3500 # density of the test mass
L = 10**(-3) # length of the plate
dt = 0.001 # time step relevant for numerical integration
timelist = np.arange(dt,(t+tau+t+dt),dt)

### functions ###
def column(matrix, i):
    return [row[i] for row in matrix]

# function for acceleration due to dipole interaction 
def acc_dd(r,m,p,theta):
    acc_dd = (3*p**2/(4*np.pi*epsilon0*16*m*r**4))*(1+math.cos(theta)**2)
    return acc_dd

# function for acceleration due to casimir interaction 
def acc_cas(r):
    acc = ((3*hbar*c)/(2*np.pi))*((epsilon-1)/(epsilon+2))*(3/(4*np.pi*sdensity*r**5))
    return acc

# function for acceleration due to magnetic field gradient
def dxacc(m,dB):
    dxacc = (g*mub*dB)/m
    return dxacc

# function of infinitesimal gravitational phase for superpositions parallel to the plate
def accphivar(m, dt, r1, r2, delx, W):
    infphase = ((2*G*m**2*dt)/hbar)*(1/np.sqrt((r1 + r2 + W)**2 + delx**2) - 1/(r1 + r2 + W))
    return infphase

# function of infinitesimal gravitational phase in the case of asymmetrically tilted superpositions
def accphivar2(m, dt, x1, x2, delx, W):
    infphase1 = ((G*m**2*dt)/hbar)*(2/np.sqrt((x1 + x2 + W)**2 + delx**2-(x2-x1)**2) - 1/(x1 + x1 + W) - 1/(x2 + x2 + W))
    return infphase1

# function of infinitesimal gravitational phase in the case of symmetrically tilted superpositions
def accphivar3(m, dt, x1, x2, delx, W):
    infphase2 = ((G*m**2*dt)/hbar)*(1/np.sqrt((x1 + x1 + W)**2 + delx**2-(x2-x1)**2) + 1/np.sqrt((x2 + x2 + W)**2 + delx**2-(x2-x1)**2) - 1/(x1 + x2 + W) - 1/(x1 + x2 + W))
    return infphase2

# function that outputs the generated gravitational entanglement phase and final positions for two superpositions initially a distance x1, x2 away from the plate
def phasefunc(m, p, dt, v1, v2, dx, x1, x2, phase):
    for i in range(len(timelist)):
        if timelist[i]<=t:
            x1 = x1 - v1*dt - 0.5*acc_dd(x1,mass,p,0)*dt**2 - 0.5*acc_cas(x1)*dt**2
            v1 = v1 + dt*acc_dd(x1,mass,p,0) + dt*acc_cas(x1)
            x2 = x2 - v2*dt - 0.5*acc_dd(x2,mass,p,0)*dt**2 - 0.5*acc_cas(x2)*dt**2
            v2 = v2 + dt*acc_dd(x2,mass,p,0) + dt*acc_cas(x2)
            dx = 0.5*dxacc(mass,flux)*timelist[i]**2
            phase = phase + accphivar(mass,dt,x1,x2,dx,W)
        elif timelist[i]<=(t+tau):
            x1 = x1 - v1*dt - 0.5*acc_dd(x1,mass,p,0)*dt**2 - 0.5*acc_cas(x1)*dt**2
            v1 = v1 + dt*acc_dd(x1,mass,p,0) + dt*acc_cas(x1)
            x2 = x2 - v2*dt - 0.5*acc_dd(x2,mass,p,0)*dt**2 - 0.5*acc_cas(x2)*dt**2
            v2 = v2 + dt*acc_dd(x2,mass,p,0) + dt*acc_cas(x2)
            dxmax = dx
            phase = phase + accphivar(mass,dt,x1,x2,dx,W)
        elif timelist[i]<=(t+2*tau):
            x1 = x1 - v1*dt - 0.5*acc_dd(x1,mass,p,0)*dt**2 - 0.5*acc_cas(x1)*dt**2
            v1 = v1 + dt*acc_dd(x1,mass,p,0) + dt*acc_cas(x1)
            x2 = x2 - v2*dt - 0.5*acc_dd(x2,mass,p,0)*dt**2 - 0.5*acc_cas(x2)*dt**2
            v2 = v2 + dt*acc_dd(x2,mass,p,0) + dt*acc_cas(x2)
            dx = dxmax - 0.5*dxacc(mass,flux)*(timelist[i]-t-tau)**2
            phase = phase + accphivar(mass,dt,x1,x2,dx,W)
            if x1<0:
                print('error')
                break
    return phase, x1, x2

# function that outputs the generated gravitational entanglement phase and final positions for two superpositions that are tilted (a)symmetrically with respect to the plate
def phasefunctilt(mass, p, dt, v1, v2, dx, x1, x2, phase_tilt, phase_tilt2):
    for i in range(len(timelist)):
        if timelist[i]<=t:
            x1 = x1 - v1*dt - 0.5*acc_dd(x1,mass,p,0)*dt**2 - 0.5*acc_cas(x1)*dt**2
            v1 = v1 + dt*acc_dd(x1,mass,p,0) + dt*acc_cas(x1)
            x2 = x2 - v2*dt - 0.5*acc_dd(x2,mass,p,0)*dt**2 - 0.5*acc_cas(x2)*dt**2
            v2 = v2 + dt*acc_dd(x2,mass,p,0) + dt*acc_cas(x2)
            dx = 0.5*dxacc(mass,flux)*timelist[i]**2
            phase_tilt = phase_tilt + accphivar2(mass,dt,x1,x2,dx,W)
            phase_tilt2 = phase_tilt2 + accphivar3(mass,dt,x1,x2,dx,W)
        elif timelist[i]<=(t+tau):
            x1 = x1 - v1*dt - 0.5*acc_dd(x1,mass,p,0)*dt**2 - 0.5*acc_cas(x1)*dt**2
            v1 = v1 + dt*acc_dd(x1,mass,p,0) + dt*acc_cas(x1)
            x2 = x2 - v2*dt - 0.5*acc_dd(x2,mass,p,0)*dt**2 - 0.5*acc_cas(x2)*dt**2
            v2 = v2 + dt*acc_dd(x2,mass,p,0) + dt*acc_cas(x2)
            dxmax = dx
            phase_tilt = phase_tilt + accphivar2(mass,dt,x1,x2,dx,W)
            phase_tilt2 = phase_tilt2 + accphivar3(mass,dt,x1,x2,dx,W)
        elif timelist[i]<=(t+2*tau):
            x1 = x1 - v1*dt - 0.5*acc_dd(x1,mass,p,0)*dt**2 - 0.5*acc_cas(x1)*dt**2
            v1 = v1 + dt*acc_dd(x1,mass,p,0) + dt*acc_cas(x1)
            x2 = x2 - v2*dt - 0.5*acc_dd(x2,mass,p,0)*dt**2 - 0.5*acc_cas(x2)*dt**2
            v2 = v2 + dt*acc_dd(x2,mass,p,0) + dt*acc_cas(x2)
            dx = dxmax - 0.5*dxacc(mass,flux)*(timelist[i]-t-tau)**2
            phase_tilt = phase_tilt + accphivar2(mass,dt,x1,x2,dx,W)
            phase_tilt2 = phase_tilt2 + accphivar3(mass,dt,x1,x2,dx,W)
            if x1<0:
                print('error')
                break
    return phase_tilt, phase_tilt2


### Dephasing ###

# via Casimir and dipole interaction with the plate due to a tilte deltad2
def dephasefunctilt(mass, p, dt, v1, v2, dx, x1, x2):
    dephase_tilt_cas = 0; dephase_tilt_dip = 0;
    for i in range(len(timelist)):
        if timelist[i]<=t:
            x1 = x1 - v1*dt - 0.5*acc_dd(x1,mass,p,0)*dt**2 - 0.5*acc_cas(x1)*dt**2
            v1 = v1 + dt*acc_dd(x1,mass,p,0) + dt*acc_cas(x1)
            x2 = x2 - v2*dt - 0.5*acc_dd(x2,mass,p,0)*dt**2 - 0.5*acc_cas(x2)*dt**2
            v2 = v2 + dt*acc_dd(x2,mass,p,0) + dt*acc_cas(x2)
            dx = 0.5*dxacc(mass,flux)*timelist[i]**2
            dephase_tilt_cas = dephase_tilt_cas + (infdephasetiltcas(mass,x1) - infdephasetiltcas(mass,x2))*dt
            dephase_tilt_dip = dephase_tilt_dip + (infdephasetiltdip(mass,x1) - infdephasetiltdip(mass,x2))*dt
        elif timelist[i]<=(t+tau):
            x1 = x1 - v1*dt - 0.5*acc_dd(x1,mass,p,0)*dt**2 - 0.5*acc_cas(x1)*dt**2
            v1 = v1 + dt*acc_dd(x1,mass,p,0) + dt*acc_cas(x1)
            x2 = x2 - v2*dt - 0.5*acc_dd(x2,mass,p,0)*dt**2 - 0.5*acc_cas(x2)*dt**2
            v2 = v2 + dt*acc_dd(x2,mass,p,0) + dt*acc_cas(x2)
            dxmax = dx
            dephase_tilt_cas = dephase_tilt_cas + (infdephasetiltcas(mass,x1) - infdephasetiltcas(mass,x2))*dt
            dephase_tilt_dip = dephase_tilt_dip + (infdephasetiltdip(mass,x1) - infdephasetiltdip(mass,x2))*dt
        elif timelist[i]<=(t+2*tau):
            x1 = x1 - v1*dt - 0.5*acc_dd(x1,mass,p,0)*dt**2 - 0.5*acc_cas(x1)*dt**2
            v1 = v1 + dt*acc_dd(x1,mass,p,0) + dt*acc_cas(x1)
            x2 = x2 - v2*dt - 0.5*acc_dd(x2,mass,p,0)*dt**2 - 0.5*acc_cas(x2)*dt**2
            v2 = v2 + dt*acc_dd(x2,mass,p,0) + dt*acc_cas(x2)
            dx = dxmax - 0.5*dxacc(mass,flux)*(timelist[i]-t-tau)**2
            dephase_tilt_cas = dephase_tilt_cas + (infdephasetiltcas(mass,x1) - infdephasetiltcas(mass,x2))*dt
            dephase_tilt_dip = dephase_tilt_dip + (infdephasetiltdip(mass,x1) - infdephasetiltdip(mass,x2))*dt
    return dephase_tilt_cas, dephase_tilt_dip

def infdephasetiltcas(mass, x):
    # For distance we assume that orientation is 0 --> 1 in the +x direction, and 0 has i.c. x(0) = d-deltad2, and 1 has x(0)=d+deltad2
    Rcubed = (3*mass)/(4*np.pi*sdensity)
    Vc = ((3*c*Rcubed)/(8*np.pi))*((epsilon-1)/(epsilon+2))*(1/(x**4))
    return Vc

def infdephasetiltdip(mass, x):
    # For distance we assume that orientation is 0 --> 1 in the +x direction, and 0 has i.c. x(0) = d-deltad2, and 1 has x(0)=d+deltad2
    Vd = (p**2/(16*np.pi*epsilon0*hbar))*(1/x**3)
    return Vd


### Loops over all the fluctuations ###

### d±u variable (fig. 9)
flux=5*10**5
mass = 10**(-14)
dt = 0.01
ulist = np.arange(0,0.7,0.01)
phaselistd = []; dxlist = []; phaselistm = []; phaselistp = [];
phaselisttiltd = []; phaselisttiltm = []; phaselisttiltp = [];
xlistm = []; xlistd = []; xlistp = [];

# loop over time for different u
for j in range(len(ulist)):
    v1 = 0; v2 = 0; dx = 0;
    x0 = 41*10**(-6)
    x1 = (41-ulist[j])*10**(-6)
    x2 = (41+ulist[j])*10**(-6)

    # delta d_1
    phasep, x1m, xtestlist = phasefunc(mass, p, dt, v1, v2, dx, x1, x1, 0)
    phasem, x1p, xtestlistw = phasefunc(mass, p, dt, v1, v2, dx, x2, x2, 0)
    phased, x1d, xtestlistww = phasefunc(mass, p, dt, v1, v2, dx, x0, x0, 0)
    
    phaselistm.append(abs(phasem))
    phaselistp.append(abs(phasep))
    phaselistd.append(abs(phased))

    xlistm.append(x1m)
    xlistp.append(x1p)
    xlistd.append(x1d)

    # delta d_2
    phasetiltm, phasetiltp = phasefunctilt(mass, p, dt, v1, v2, dx, x1, x2, 0, 0)
    phasetiltd, phasetiltd2 = phasefunctilt(mass, p, dt, v1, v2, dx, x0, x0, 0, 0)
    
    phaselisttiltm.append(abs(phasetiltm))
    phaselisttiltp.append(abs(phasetiltp))
    phaselisttiltd.append(abs(phasetiltd))
    
    j=j+1

### Magnetic flux variation (fig. 10)
dfluxlist = np.arange(0,4*10**4,0.5*10**2)
phaselist = []; dxlist = []; phaselistm = []; dxlistm = []; phaselistp = []; dxlistp = []
flux = 5*10**5
mass = 10**(-14)

# loop over time for different fluxes
for j in range(len(dfluxlist)):
    v = 0; vy = 0; dx = 0; x = 41*10**(-6); phase = 0; phasem = 0; phasep = 0;
    for i in range(len(timelist)):
        if timelist[i]<=t:
            x = x - v*dt - 0.5*accdd(x,mass,p,0)*dt**2 - 0.5*acc(x)*dt**2
            v = v + dt*accdd(x,mass,p,0) + dt*acc(x)
            dx = 0.5*dxacc(mass,flux)*timelist[i]**2
            dxm = 0.5*dxacc(mass,flux-dfluxlist[j])*timelist[i]**2
            dxp = 0.5*dxacc(mass,flux+dfluxlist[j])*timelist[i]**2
            phase = phase + accphi(mass,dt,x,dx,W)
            phasem = phasem + accphi(mass,dt,x,dxm,W)
            phasep = phasep + accphi(mass,dt,x,dxp,W)
        elif timelist[i]<=(t+tau):
            x = x - v*dt - 0.5*accdd(x,mass,p,0)*dt**2 - 0.5*acc(x)*dt**2
            v = v + dt*accdd(x,mass,p,0) + dt*acc(x)
            phase = phase + accphi(mass,dt,x,dx,W)
            phasem = phasem + accphi(mass,dt,x,dxm,W)
            phasep = phasep + accphi(mass,dt,x,dxp,W)
            dxmax = dx; dxmaxp = dxp; dxmaxm = dxm
        elif timelist[i]<=(t+2*tau):
            x = x - v*dt - 0.5*accdd(x,mass,p,0)*dt**2 - 0.5*acc(x)*dt**2
            v = v + dt*accdd(x,mass,p,0) + dt*acc(x)
            dx = dxmax - 0.5*dxacc(mass,flux)*(timelist[i]-t-tau)**2
            dxm = dxmaxm - 0.5*dxacc(mass,flux-dfluxlist[j])*(timelist[i]-t-tau)**2
            dxp = dxmaxp - 0.5*dxacc(mass,flux+dfluxlist[j])*(timelist[i]-t-tau)**2
            phase = phase + accphi(mass,dt,x,dx,W)
            phasem = phasem + accphi(mass,dt,x,dxm,W)
            phasep = phasep + accphi(mass,dt,x,dxp,W)
    dxlist.append(dxmax*10**6)
    dxlistm.append(dxmaxm*10**(6))
    dxlistp.append(dxmaxp*10**(6))
    phaselist.append(abs(phase))
    phaselistm.append(abs(phasem))
    phaselistp.append(abs(phasep))
    j=j+1

### Fluctuations in the dipole orientation (fig. 11)
#(assuming they are aligned)
dthetalist = np.arange(0,np.pi/2,0.005)
phaselist = []; dxlist = []; phaselistm = []; dxlistm = []; phaselistp = []; dxlistp = []
timelist = np.arange(dt,(t+tau+t+dt),dt)
flux = 5*10**5
mass = 10**(-14)

# loop over time for different angles
for j in range(len(dthetalist)):
    v = 0; vy = 0; dx = 0; x = 41*10**(-6); phase = 0; phasem = 0; phasep = 0; theta=dthetalist[j];
    for i in range(len(timelist)):
        if timelist[i]<=t:
            x = x - v*dt - 0.5*accdd(x,mass,p,theta)*dt**2 - 0.5*acc(x)*dt**2
            v = v + dt*accdd(x,mass,p,theta) + dt*acc(x)
            dx = 0.5*dxacc(mass,flux)*timelist[i]**2
            phase = phase + accphi(mass,dt,x,dx,W)
        elif timelist[i]<=(t+tau):
            x = x - v*dt - 0.5*accdd(x,mass,p,theta)*dt**2 - 0.5*acc(x)*dt**2
            v = v + dt*accdd(x,mass,p,theta) + dt*acc(x)
            phase = phase + accphi(mass,dt,x,dx,W)
            dxmax = dx;
        elif timelist[i]<=(t+2*tau):
            x = x - v*dt - 0.5*accdd(x,mass,p,theta)*dt**2 - 0.5*acc(x)*dt**2
            v = v + dt*accdd(x,mass,p,theta) + dt*acc(x)
            dx = dxmax - 0.5*dxacc(mass,flux)*(timelist[i]-t-tau)**2
            phase = phase + accphi(mass,dt,x,dx,W)
    dxlist.append(dxmax*10**6)
    phaselist.append(abs(phase))
    j=j+1

for ind in range(len(phaselist)):
    if phaselist[ind]<(0.88*phaselist[0]):
        print('max value theta:',dthetalist[ind-1])
        print('phase:',phaselist[ind-1])
        break


### Functions for figs on deflection (fig. 12)
# function for deflection plate
def deflection(a,m,r1,r2,theta1,theta2):
    defl=((L-2*a)**2/(16*W**3*E))*(1+4*a/L)*(np.abs(acc(r1) - acc(r2))+np.abs(accdd(r1,m,p,theta1)-accdd(r2,m,p,theta2)))*m
    return defl

# function to find final positions
def posfunc(m, p, dt, v1, v2, x1, x2, theta1, theta2):
    xtestlist=[]; 
    for i in range(len(timelist)):
        if timelist[i]<=t:
            x1 = x1 - v1*dt - 0.5*accdd(x1,mass,p,theta1)*dt**2 - 0.5*acc(x1)*dt**2
            v1 = v1 + dt*accdd(x1,mass,p,theta1) + dt*acc(x1)
            x2 = x2 - v2*dt - 0.5*accdd(x2,mass,p,theta2)*dt**2 - 0.5*acc(x2)*dt**2
            v2 = v2 + dt*accdd(x2,mass,p,theta2) + dt*acc(x2)
            xtestlist.append(x1)
        elif timelist[i]<=(t+tau):
            x1 = x1 - v1*dt - 0.5*accdd(x1,mass,p,theta1)*dt**2 - 0.5*acc(x1)*dt**2
            v1 = v1 + dt*accdd(x1,mass,p,theta1) + dt*acc(x1)
            x2 = x2 - v2*dt - 0.5*accdd(x2,mass,p,theta2)*dt**2 - 0.5*acc(x2)*dt**2
            v2 = v2 + dt*accdd(x2,mass,p,theta2) + dt*acc(x2)
            xtestlist.append(x1)
        elif timelist[i]<=(t+2*tau):
            x1 = x1 - v1*dt - 0.5*accdd(x1,mass,p,theta1)*dt**2 - 0.5*acc(x1)*dt**2
            v1 = v1 + dt*accdd(x1,mass,p,theta1) + dt*acc(x1)
            x2 = x2 - v2*dt - 0.5*accdd(x2,mass,p,theta2)*dt**2 - 0.5*acc(x2)*dt**2
            v2 = v2 + dt*accdd(x2,mass,p,theta2) + dt*acc(x2)
            xtestlist.append(x1)
    return x1, x2

astep = 10**(-5)
alist = np.arange(0,L/2+10**(-5),astep)
flux = 5*10**5
mass = 10**(-14)
theta = 0.17*np.pi#0.15*np.pi
deltad1 = 0.48#0.42
defl_list_theta = []; defl_list = [];

# plot loop
for j in range(len(alist)):
    a = alist[j]

    # delta d1
    v1 = 0; v2 = 0;
    x1 = (41-deltad1)*10**(-6)
    x2 = (41+deltad1)*10**(-6)
    r1, r2 = posfunc(mass, p, dt, v1, v2, x1, x2, 0, 0)
    defl_list.append(deflection(a,mass,r1,r2,0,0))

    #delta theta
    v0 = 0; x0 = (41)*10**(-6);
    r1, r2 = posfunc(mass, p, dt, v0, v0, x0, x0, 0, theta)
    defl_list_theta.append(deflection(a,mass,r1,r2,0,theta))

    j=j+1
