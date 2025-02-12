from analcisprop.constants import *
import numpy as np
from numpy import linalg as LA
from analcisprop.utils import spc
from scipy.optimize import newton

def wrap_to_2pi(angle):
    """Wrap angle to the interval [0, 2π)."""
    return np.mod(angle, 2*np.pi)

def M2E(M, e):
    """
    Calculate eccentric anomaly given mean anomaly
    """
    def kepler_equation(E, M, e):
        return E - e * np.sin(E) - M
    E = newton(kepler_equation, M, args=(M, e))
    return E


def E2M(E, e):
    """
    Convert Eccentric Anomaly (E) to Mean Anomaly (M) using Kepler's equation.
    """
    return np.mod( E - e * np.sin(E) , 2*np.pi)


def ic2par(sol, mu):
     """
     This function converts any momentary value of the state vector z 
     of an object orbiting the earth into the corresponding osculating
     Keplerian elements.

     INPUT: cartesian state vector z
     OUTPUT: 
        a: semi-major axis   
        e: orbital eccentricity  
        i: inclination
        M: mean anomaly
        omega: argument of perigee
        RAAN: right ascension of the ascending node 
     """

     small = 1e-12

     # spit input in position and velocity vectors
     rv = sol[:3]
     vv = sol[3:]
     mag_rv = LA.norm(rv)
     mag_vv = LA.norm(vv)
     
     # orbital momentum
     hv = np.cross(rv, vv)
     mag_hv = LA.norm(hv)

     #eccentricity
     ev = np.cross(vv, hv)/mu - rv/mag_rv
     ecc = LA.norm(ev)

     # vector pointing towards the ascending node
     nv = np.array( [-hv[1], hv[0], 0] )
     mag_nv = LA.norm(nv)

     # true anomaly
     if np.dot(rv, vv) >= 0:
          nu = np.arccos( np.dot(ev, rv)/( ecc*mag_rv ) )
     else:
         nu = 2*np.pi - np.arccos( np.dot(ev, rv)/( ecc*mag_rv ) )
     
     # inclination
     inc = np.arccos(hv[-1]/mag_hv)

     # RAAN
     if nv[1] >=0:
          OM = np.arccos(nv[0]/mag_nv)
     else:
          OM = 2*np.pi - np.arccos(nv[0]/mag_nv)
     
     # argument of perigee
     if ev[-1] >=0:
          om = np.arccos( np.dot(nv, ev)/( mag_nv*ecc ) )
     else:
          om = 2*np.pi - np.arccos( np.dot(nv, ev)/( mag_nv*ecc ) )
     
     # eccentric and mean anomaly
     den =  1+ ecc*np.cos(nu)
     eccan = np.arctan2( ( np.sqrt(1-ecc**2)*np.sin(nu) )/den, ( ecc + np.cos(nu) )/den )

     # semi-major axis
     a = 1/( 2/mag_rv - (mag_vv**2)/mu )

     return [a, ecc, inc, wrap_to_2pi(OM), wrap_to_2pi(om), wrap_to_2pi(eccan)]

def par2ic( kep, mu ):
     [a, ecc, inc, OM, om, eccan] = kep

     # true anomaly
     dennu = 1 - ecc*np.cos(eccan)
     sinnu = ( np.sqrt(1 - ecc**2)*np.sin(eccan) )/dennu
     cosnu = (np.cos(eccan) - ecc )/dennu
     nu = np.arctan2( sinnu, cosnu ) 

     # position vector
     mag_rv = a*( 1 - ecc*np.cos(eccan) )
     orv = mag_rv*np.array([np.cos(nu), np.sin(nu), 0])
     ovv = (np.sqrt(mu*a))/mag_rv * np.array([-np.sin(eccan), np.sqrt(1-ecc**2)*np.cos(eccan), 0 ])
     orx, ory = orv[0], orv[1] 
     ovx, ovy = ovv[0], ovv[1] 

     com, som = np.cos(om), np.sin(om)
     cOM, sOM = np.cos(OM), np.sin(OM)
     ci, si = np.cos(inc), np.sin(inc)

     rv = np.array([
          orx*(com*cOM - som*ci*sOM) - ory*(som*cOM + com*ci*sOM),
          orx*(com*sOM + som*ci*cOM) + ory*(com*ci*cOM - som*sOM),
          orx*(som*si) + ory*(com*si) ])
     
     vv = np.array([
          ovx*(com*cOM - som*ci*sOM) - ovy*(som*cOM + com*ci*sOM),
          ovx*(com*sOM + som*ci*cOM) + ovy*(com*ci*cOM - som*sOM),
          ovx*(som*si) + ovy*(com*si) ])

     return np.array([rv, vv]).flatten()

# Keplerian elements to equinoxial.
def kep2equinox(kep):
    a, e, i, OM, om, M = kep
    h = e*np.cos(OM+om)
    k = e*np.sin(OM+om)
    p = np.sin(i/2)*np.cos(OM)
    q = np.sin(i/2)*np.sin(OM)
    lM = M + OM + om

    return [a, h, k, p, q, lM]

# Equinoxial elements to Keplerian.
def equinox2kep(equinox):

    a, h, k, p, q, lM = equinox
    e = np.sqrt(h*h + k*k);
    i = 2.0*np.arcsin(np.sqrt(p*p + q*q));
    OM = np.arctan2(q,p)
    OM_p_om = np.arctan2(k,h)
    om = OM_p_om - OM
    M = lM - OM_p_om

    return [a, e, i, wrap_to_2pi(OM), wrap_to_2pi(om), wrap_to_2pi(M)]

# Delaunay to Keplerian
def del2kep(DEL, L):
    """ 
    From  Delaunay variables to Keplerian elements

    INPUT : L, G, H, g, h
    OUTPUT: a, e, irad, g, h
    
    [a, e, irad, h, g] = DTK( [L, G, H, g, h] )
    """
    G,H,g,h  = DEL
    a = (L**2)/GML
    e = np.sqrt( 1 - (G**2/L**2) )
    irad = np.arccos( H/G )
    return [a, e, irad, wrap_to_2pi(h), wrap_to_2pi(g)]

# Keplerian to Delaunay
def kep2del(kep):
    """
    From Keplerian elements to Delaunay variables

    INPUT : akm, e, irad, om, OM
    OUTPUT: L, G, H, OM, om 
    """
    akm, e, irad, om, OM = kep
    L = np.sqrt( (akm)*GML )
    G = L*np.sqrt(1-e**2)
    H = G*np.cos(irad)
    return [L, G, H, wrap_to_2pi(OM), wrap_to_2pi(om)]

def kep2spcvars(kep):
    akm, ecc0, inc0, OM0, om0, M0 = kep
    ecc_an = M2E(M0, ecc0)
    true_an = 2 * np.arctan(np.sqrt((1 + ecc0) / (1 - ecc0)) * np.tan(ecc_an / 2))
    r = (akm * (1 - ecc0**2)) / (1 + ecc0 * np.cos(true_an))
    n = np.sqrt(GML / akm**3)
    cih = np.cos(inc0/2)
    sih = np.sin(inc0/2)
    eta = np.sqrt(1 - ecc0**2)
    etap1 = eta + 1
    lw = om0 + OM0
    lu = lw + ecc_an
    phi = true_an - M0
    lf = true_an + OM0 + om0
    return [r, n, cih, sih, eta, etap1, lw, lu, phi, lf]

def mod2del(mod, L):
    P, Q, p, q = mod
    G = L-P
    H = G-Q
    h = -q 
    g = -p+q
    return [G, H, g, h]

def del2mod(DEL, L):
    G, H, g, h = DEL
    P = L-G
    Q = G-H
    q = -h
    p = q-g 
    return [P, Q, p, q]

def convert(seconds):
    """converts a number of seconds into hours:minutes:seconds """
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hours, minutes, seconds)

def mean2osc(mean_kep, t):
    [a, ecc, inc, OM, om, mean_an] = mean_kep
    mean_eq = kep2equinox(mean_kep)
    [xE, yE, zE] = earth_position(t)
    rE = np.sqrt(xE**2 + yE**2 + zE**2)
    [r, n, cih, sih, eta, etap1, lw, lu, phi, lf] = kep2spcvars(mean_kep)
    internal_vals = [a, ecc, n, cih, sih, eta, etap1, phi, RL, OM, lw, lf]
    external_vals = [rE, xE, yE, zE, r, a, ecc, n, cih, sih, eta, etap1, OM, lw, lu]

    med_eq = spc.mean2med(mean_eq, internal_vals, verse=1)
    osc_eq = spc.med2osc(med_eq, internal_vals, external_vals, verse=1)
    osc_kep = equinox2kep(osc_eq)

    return osc_kep

def osc2mean(osc_kep, t):
    [a, ecc, inc, OM, om, mean_an] = osc_kep
    osc_eq = kep2equinox(osc_kep)
    [xE, yE, zE] = earth_position(t)
    rE = np.sqrt(xE**2 + yE**2 + zE**2)
    [r, n, cih, sih, eta, etap1, lw, lu, phi, lf] = kep2spcvars(osc_kep)
    internal_vals = [a, ecc, n, cih, sih, eta, etap1, phi, RL, OM, lw, lf]
    external_vals = [rE, xE, yE, zE, r, a, ecc, n, cih, sih, eta, etap1, OM, lw, lu]

    med_eq = spc.med2osc(osc_eq, internal_vals, external_vals, verse=-1)
    mean_eq = spc.mean2med(med_eq, internal_vals, verse=-1)
    mean_kep = equinox2kep(mean_eq)

    return mean_kep