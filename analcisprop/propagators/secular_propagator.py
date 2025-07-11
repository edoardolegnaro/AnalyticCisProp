import numpy as np
from scipy.integrate import solve_ivp

from analcisprop.constants import GML, AxV, AyV, AzV, BxV, ByV, BzV, wxV, wyV, wzV
from analcisprop.propagators.secular_equations.dh_dt import dh_dt
from analcisprop.propagators.secular_equations.dk_dt import dk_dt
from analcisprop.propagators.secular_equations.dlM_dt import dlM_dt
from analcisprop.propagators.secular_equations.dp_dt import dp_dt
from analcisprop.propagators.secular_equations.dq_dt import dq_dt
from analcisprop.utils import variable_changes as vc


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


def propagate(kep_elements, tmax, dt, **options):
    """
    Propagates Keplerian orbital elements using the secular propagator.

    Args:
        kep_elements: Initial Keplerian elements [a, e, i, OM, om, M] where:
                     - a: semi-major axis (km)
                     - e: eccentricity
                     - i: inclination (rad)
                     - OM: longitude of ascending node (rad)
                     - om: argument of periapsis (rad)
                     - M: mean anomaly (rad)
        tmax: Maximum integration time (s)
        dt: Time step for the output grid (s)
        **options: Additional keyword arguments passed to scipy.integrate.solve_ivp.

    Returns:
        dict: Dictionary containing:
            - 't': Time grid (s)
            - 'keplerian': List of Keplerian elements at each time step
                          Each element is [a, e, i, OM, om, M]
            - 'solution': Raw scipy solution object (in equinoctial coordinates)
    """
    # Convert Keplerian to equinoctial coordinates
    ic0_eq = np.array(vc.kep2equinox(kep_elements))

    # Set default solver options
    default_options = {"method": "LSODA", "rtol": 1e-8, "atol": 1e-8}

    # Update defaults with user-provided options
    solver_options = {**default_options, **options}

    # Create time grid using dt as step size
    t_eval = np.arange(0, tmax + dt, dt)

    solver_options["t_eval"] = t_eval

    t_span = (0, tmax)

    # Propagate in equinoctial coordinates
    sec_sol_eq = solve_ivp(VF, t_span, ic0_eq, **solver_options)

    # Convert results back to Keplerian elements
    sol_sec_kep_list = []
    for j in range(sec_sol_eq.y.shape[1]):
        equinoctial_state = sec_sol_eq.y[:, j]
        keplerian_state = vc.equinox2kep(equinoctial_state)
        sol_sec_kep_list.append(keplerian_state)

    return {"t": sec_sol_eq.t, "keplerian": sol_sec_kep_list, "equinox": sec_sol_eq}


def propagate_equinoctial(y0, t_span, **options):
    """
    Legacy function: Propagates the state vector y0 over the time span t_span using the VF function.

    This function maintains backward compatibility with the original interface.
    For new code, consider using the propagate() function which accepts Keplerian elements directly.

    Args:
        y0: Initial state vector [a, h, k, p, q, lM] in equinoctial coordinates.
        t_span: Tuple (t_start, t_end) specifying the integration interval.
        **options: Additional keyword arguments passed to scipy.integrate.solve_ivp.

    Returns:
        The solution object returned by scipy.integrate.solve_ivp.
    """
    sol = solve_ivp(VF, t_span, y0, **options)
    return sol
