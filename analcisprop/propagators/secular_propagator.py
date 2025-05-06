from analcisprop.propagators.secular_equations.dh_dt import dh_dt
from analcisprop.propagators.secular_equations.dk_dt import dk_dt
from analcisprop.propagators.secular_equations.dp_dt import dp_dt
from analcisprop.propagators.secular_equations.dq_dt import dq_dt
from analcisprop.propagators.secular_equations.dlM_dt import dlM_dt
from analcisprop.constants import AxV, BxV, wxV, AyV, ByV, wyV, AzV, BzV, wzV, GML
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

def VF(t, y):
    """
    y is the state vector in equinoctical coordinates.
    y = [a, h, k, p, q, lM]
    """
    akm, h, k, p, q, lM = y
    # Earth Contribution
    xenum_sum = 0.0
    for j in range(len(AxV)):
        xenum_sum += AxV[j] * np.cos(wxV[j] * t) + BxV[j] * np.sin(wxV[j] * t)
    xenum = 382469.63 + xenum_sum

    yenum_sum = 0.0
    for j in range(len(AyV)):
        yenum_sum += AyV[j] * np.cos(wyV[j] * t) + ByV[j] * np.sin(wyV[j] * t)
    yenum = yenum_sum

    zenum_sum = 0.0
    for j in range(len(AzV)):
        zenum_sum += AzV[j] * np.cos(wzV[j] * t) + BzV[j] * np.sin(wzV[j] * t)
    zenum = zenum_sum

    renum = np.sqrt(xenum**2 + yenum**2 + zenum**2)

    OM = np.arctan2(q, p)
    lw = np.arctan2(k, h)

    sih = np.sqrt(q**2 + p**2)
    if sih * sih >= 1.0:
        cih = 0.0
    else:
        cih = np.sqrt(1 - sih**2)

    ecc_sq = h**2 + k**2
    if ecc_sq >= 1.0:
        eta = 0.0
        ecc = 1.0
    else:
        ecc = np.sqrt(ecc_sq)
        eta = np.sqrt(1 - ecc_sq)

    n = np.sqrt(GML / akm**3)

    hdot = dh_dt(akm, OM, n, ecc, sih, cih, eta, lw, xenum, yenum, zenum, renum)
    kdot = dk_dt(akm, OM, n, ecc, sih, cih, eta, lw, xenum, yenum, zenum, renum)
    pdot = dp_dt(akm, OM, n, ecc, sih, cih, eta, lw, xenum, yenum, zenum, renum)
    qdot = dq_dt(akm, OM, n, ecc, sih, cih, eta, lw, xenum, yenum, zenum, renum)
    lMdot = dlM_dt(akm, OM, n, ecc, sih, cih, eta, lw, xenum, yenum, zenum, renum)

    out = np.empty(6, dtype=np.float64)
    out[0] = 0.0
    out[1] = hdot
    out[2] = kdot
    out[3] = qdot
    out[4] = pdot
    out[5] = lMdot
    return out

def propagate(y0, t_span, **options):
    """
    Propagates the state vector y0 over the time span t_span using the VF function.

    Args:
        y0: Initial state vector [a, h, k, p, q, lM].
        t_span: Tuple (t_start, t_end) specifying the integration interval.
        **options: Additional keyword arguments passed to scipy.integrate.solve_ivp.

    Returns:
        The solution object returned by scipy.integrate.solve_ivp.
    """
    sol = solve_ivp(VF, t_span, y0, **options)
    return sol