import numpy as np
import importlib.resources as resources
from analcisprop.constants import Cbar, Sbar, GME
import os

spc_dir = 'spc'

def load_spc_3b(file):
    theory = []
    for line in file:
        # Skip empty lines or comments
        if not line.strip() or line.startswith('#'):
            continue
        values = line.strip().split()
        if len(values) != 17:
            # Handle error or skip line
            continue
        
        num_coeff = float(values[0])
        r3b_exp    = float(values[1])
        x3b_exp    = float(values[2])
        y3b_exp    = float(values[3])
        z3b_exp    = float(values[4])
        r_exp      = float(values[5])
        a_exp      = float(values[6])
        e_exp      = float(values[7])
        n_exp      = float(values[8])
        cih_exp    = float(values[9])
        sih_exp    = float(values[10])
        eta_exp    = float(values[11])
        etap1_exp  = float(values[12])
        cos_sin_flag = int(values[13])  # 0: use cosine, 1: use sine

        RAAN_mult = float(values[14])
        lw_mult   = float(values[15])
        lu_mult   = float(values[16])

        row = [num_coeff, r3b_exp, x3b_exp, y3b_exp, z3b_exp, r_exp, a_exp, e_exp, n_exp, 
                cih_exp, sih_exp, eta_exp, etap1_exp, cos_sin_flag, RAAN_mult, lw_mult, lu_mult]
        theory.append(row)
    return theory

def load_spc_internal(file):
    internal_terms = []
    for line in file:
        # Skip empty lines or comments
        if not line.strip() or line.startswith('#'):
            continue
        values = line.strip().split()
        if len(values) != 17:
            # Handle error or skip line
            continue
        
        cos_sin_flag = int(values[0])  # 0: Cnm, 1: Snm
        n = int(values[1])
        m = int(values[2])
        coeff = float(values[3])
        a_exp = float(values[4])
        e_exp = float(values[5])
        n_exp = float(values[6])
        cih_exp = float(values[7])
        sih_exp = float(values[8])
        eta_exp = float(values[9])
        etap1_exp = float(values[10])
        phi_exp = float(values[11])
        RL_exp = float(values[12])
        trig_flag = int(values[13])  # 0: cosine, 1: sine
        h_mult = float(values[14])
        lw_mult = float(values[15])
        lf_mult = float(values[16])

        row = [
            cos_sin_flag, n, m, coeff, a_exp, e_exp, n_exp, cih_exp, sih_exp,
            eta_exp, etap1_exp, phi_exp, RL_exp, trig_flag, h_mult, lw_mult, lf_mult
        ]
        internal_terms.append(row)

    return internal_terms

def reconstruct_3bp(coeff_list, GM3b, r3b, x3b, y3b, z3b, r, a, e, n, cih, sih, eta, etap1, h, lw, lu):
    total_sum = 0

    for cl in coeff_list:
        cct = cl[0]
        cr = r3b ** cl[1]
        cx = x3b ** cl[2]
        cy = y3b ** cl[3]
        cz = z3b ** cl[4]
        crs = r ** cl[5]
        ca = a ** cl[6]
        ce = e ** cl[7]
        cn = n ** cl[8]
        ccih = cih ** cl[9]
        csih = sih ** cl[10]
        ceta = eta ** cl[11]
        cetap1 = etap1 ** cl[12]

        # Determine whether to use cosine or sine
        if cl[13] == 0:
            ctrig = np.cos(cl[14] * h + cl[15] * lw + cl[16] * lu)
        else:
            ctrig = np.sin(cl[14] * h + cl[15] * lw + cl[16] * lu)

        cterm = cct * cr * cx * cy * cz * crs * ca * ce * cn * ccih * csih * ceta * cetap1 * ctrig
        total_sum += cterm

    return total_sum * GM3b

def reconstruct_internal(coeff_list, a, e, n, cih, sih, eta, etap1, phi, RL, h, lw, lf, verse):
    """ Compute the internal terms sum, with optional handling for J2² terms. """
    total_sum = 0

    for cl in coeff_list:
        n_val, m_val = int(cl[1]), int(cl[2])  # Extract n and m
        
        # Choose C or S based on the flag
        ccsnm = float(verse)*( Cbar(n_val, m_val) if cl[0] == 0 else Sbar(n_val, m_val))

        cct = cl[3]
        ca = a ** cl[4]
        ce = e ** cl[5]
        cn = n ** cl[6]
        ccih = cih ** cl[7]
        csih = sih ** cl[8]
        ceta = eta ** cl[9]
        cetap1 = etap1 ** cl[10]
        cphi = phi ** cl[11]
        crl = RL ** cl[12]

        # Compute cosine or sine based on cos_sin_flag
        if cl[13] == 0:
            ctrig = np.cos(cl[14] * h + cl[15] * lw + cl[16] * lf)
        else:
            ctrig = np.sin(cl[14] * h + cl[15] * lw + cl[16] * lf)

        # Compute the term
        cterm = cct * ccsnm * ca * ce * cn * ccih * csih * ceta * cetap1 * cphi * crl * ctrig

        # Sum the term
        total_sum += cterm

    return total_sum

def spc_moon(var, internal_vals, type_spc, verse):
    akm, ecc0, n, cih, sih, eta, etap1, phi, RL, OM0, lw, lf = internal_vals
    resource_package = f"analcisprop.spc.internal.{type_spc}"
    resource_name = f"{var}.txt"
    path = os.path.join(spc_dir, 'internal', type_spc, f'{var}.txt')
    with resources.open_text(resource_package, resource_name) as file:
        cf_list = load_spc_internal(file)
    return reconstruct_internal(cf_list, akm, ecc0, n, cih, sih, eta, etap1, phi, RL, OM0, lw, lf, verse)

def spc_3b(var, external_vals, type_spc, verse):
    rE, xE, yE, zE, r, akm, ecc0, n, cih, sih, eta, etap1, OM0, lw, lu = external_vals
    resource_package = f"analcisprop.spc.external.P2.{type_spc}"
    resource_name = f"{var}.txt"
    with resources.open_text(resource_package, resource_name) as file:
        cf_list = load_spc_3b(file)
    return reconstruct_3bp(cf_list, verse*GME, rE, xE, yE, zE, r, akm, ecc0, n, cih, sih, eta, etap1, OM0, lw, lu)


keys = ['a', 'h', 'k', 'p', 'q', 'lM']

def mean2med(mean_eq, internal_vals, verse):
    """
    verse = +1: Applies forward transformation (mean → medium), so when using mean2osc()
    verse = -1: Applies inverse transformation (medium → mean), so when using osc2mean()
    """
    
    corrections = {
        key: spc_moon(key, internal_vals, 'mean2med', verse)
        for key in keys
        }
    
    return [elem + corrections[key]
            for elem, key in zip(mean_eq, keys)]


def med2osc(med_eq, internal_vals, external_vals, verse):

    internal_corr = {
        key: spc_moon(key, internal_vals, 'med2osc', verse)
        for key in keys
    }
    
    external_corr = {
        key: spc_3b(key, external_vals, 'firstorder', verse)
        for key in keys
    }

    return [elem + internal_corr[key] + external_corr[key] 
            for elem, key in zip(med_eq, keys)]




