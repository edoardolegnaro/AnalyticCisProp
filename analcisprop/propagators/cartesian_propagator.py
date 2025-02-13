import heyoka as hk
from analcisprop.constants import *


x, y, z, px, py, pz = hk.make_vars("x", "y", "z", "px", "py", "pz")
t = hk.time

r = hk.sqrt(x**2 + y**2 + z**2)

# Keplerian Component
Hkep = ( px**2 + py**2 + pz**2 )/2 - GML/r

# Non Inertial (centrifugal) Component
w3 = 0.229968
HNI = - (py*x - px*y)*w3

# Moon Internal Potential
VMoon = (2.2467009773535616e16*r**16 + r**14*(1.0937313388684064e22 - 3.2833934984417137e19*x - 4.1123514979852104e16*x**2 - 6.792102237082846e18*y - 2.627751434075474e16*y**2 - 2.4380849916185134e18*z) + r**2*(-1.6390097088624157e40 - 1.4822255084781608e36*z)*z**7 + 7.739768069628074e39*z**9 + r**4*z**5*(1.147306796203691e40 + (2.766820949159234e36 - 1.799449543373165e33*x)*z - 1.0183098766282603e33*z**2) + r**6*z**3*(-2.941812297958182e39 + (-1.5962428552841737e36 + 2.0762879346613438e33*x)*z + 1.6449621083994974e33*z**2 - 2.00501955077752e29*z**3) + r**12*(-8.6797383150542e28 + 4.104241873052142e19*x**3 - 1.2152570431871182e22*y**2 + 8.490127796353556e18*y**3 + x**2*(-1.2152570431871182e22 + 8.490127796353556e18*y) + x*(4.104241873052142e19*y**2 - 1.4288261193445725e22*z) + 4.0634749860308557e18*z**3) + r**8*(-2.9022597368803157e35*x**2 + 5.662603458167302e32*x**3 - 2.9022597368803157e35*y**2 + 5.662603458167302e32*x*y**2 + 2.005781112244215e38*z - 7.477100492724988e32*z**3 + 2.7341175692420734e29*z**4) + r**10*(2.821641410855862e35 + 9.11372523080691e28*x**2 + 9.11372523080691e28*y**2 + 8.307889436361097e31*z - 1.4177998837183047e22*z**4 + x*(-5.452877404161105e32 + 3.333927611804003e22*z**3)))/r**19

# Earth Contribution
xE = 382469.63+ np.sum([AxV[j] * hk.cos(wxV[j] * t) + BxV[j] * hk.sin(wxV[j] * t) for j in range(len(AxV))], axis=0)
yE = np.sum([AyV[j] * hk.cos(wyV[j] * t) + ByV[j] * hk.sin(wyV[j] * t) for j in range(len(AyV))], axis=0)
zE = np.sum([AzV[j] * hk.cos(wzV[j] * t) + BzV[j] * hk.sin(wzV[j] * t) for j in range(len(AzV))], axis=0)

rE = hk.sqrt( xE**2 + yE**2 + zE**2 )
rrE = x*xE + y*yE + z*zE

VEarth = -GME*( 1/hk.sqrt( r**2 + rE**2 -2*rrE ) - rrE/rE**3 )

# Cartesian Hamiltonian
Ham = Hkep + HNI + VMoon + VEarth

# Equations of Motion
ic0 = [2500., 456., 975.,
       1265., 345., 2356. ]

xdot = hk.diff(Ham, px)
ydot = hk.diff(Ham, py)
zdot = hk.diff(Ham, pz)
pxdot =  - hk.diff(Ham, x)
pydot =  - hk.diff(Ham, y)
pzdot =  - hk.diff(Ham, z)


ta = hk.taylor_adaptive(
    [(x, xdot), (y, ydot), (z, zdot),
     (px, pxdot), (py, pydot), (pz, pzdot)],
     ic0,
     compact_mode = True
)

print("\nHeyoka Taylor integrator:\n", ta)
print("\nTaylor order:\n", len(ta.decomposition) - 12)

def propagate(tmax, dt, ic):
    ta.time = 0
    ta.state[:] = ic
    tgrid = np.linspace(0.0, tmax, int(1 + tmax/dt), endpoint = True)
    out = ta.propagate_grid(tgrid)
    return out[5]