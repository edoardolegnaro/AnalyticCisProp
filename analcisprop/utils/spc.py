import numpy as np
import importlib.resources as resources
from analcisprop.constants import Cbar, Sbar, GME
import os
import functools

spc_dir = 'spc'

# Cache for loaded coefficients
_coefficient_cache = {}

@functools.lru_cache(maxsize=32)
def load_spc_file(resource_package, resource_name):
    """Load SPC coefficients from file with caching"""
    with resources.open_text(resource_package, resource_name) as file:
        if "internal" in resource_package:
            return load_spc_internal(file)
        else:
            return load_spc_3b(file)

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
    """Reconstruction of 3-body perturbation terms"""
    # Pre-compute powers that are used multiple times
    powers = {
        'r3b': r3b, 'x3b': x3b, 'y3b': y3b, 'z3b': z3b, 'r': r,
        'a': a, 'e': e, 'n': n, 'cih': cih, 'sih': sih,
        'eta': eta, 'etap1': etap1
    }
    
    coeff_array = np.array(coeff_list)
    
    # Extract columns from coefficient array
    ccts = coeff_array[:, 0]
    exponents = coeff_array[:, 1:13]  # All exponents in columns 1-12
    cos_sin_flags = coeff_array[:, 13].astype(int)
    angle_mults = coeff_array[:, 14:17]  # Angle multipliers
    
    terms = ccts.copy()
    
    # Multiply by all power terms
    for i, power_key in enumerate(['r3b', 'x3b', 'y3b', 'z3b', 'r', 'a', 'e', 'n', 
                                  'cih', 'sih', 'eta', 'etap1']):
        # Use broadcasting to compute all powers at once
        terms *= powers[power_key] ** exponents[:, i]
    
    # Compute angles
    angles = angle_mults[:, 0] * h + angle_mults[:, 1] * lw + angle_mults[:, 2] * lu
    
    # Apply cosine or sine based on flag
    cos_terms = np.cos(angles)
    sin_terms = np.sin(angles)
    trig_terms = np.where(cos_sin_flags == 0, cos_terms, sin_terms)
    
    # Final multiplication and sum
    return GM3b * np.sum(terms * trig_terms)

def reconstruct_internal(coeff_list, a, e, n, cih, sih, eta, etap1, phi, RL, h, lw, lf, verse):
    """Reconstruction of internal perturbation terms"""
    # Pre-compute common values
    powers = {
        'a': a, 'e': e, 'n': n, 'cih': cih, 'sih': sih,
        'eta': eta, 'etap1': etap1, 'phi': phi, 'RL': RL
    }
    
    coeff_array = np.array(coeff_list)
    
    # Extract all needed values as arrays
    cs_flags = coeff_array[:, 0].astype(int)
    n_vals = coeff_array[:, 1].astype(int)
    m_vals = coeff_array[:, 2].astype(int)
    coeffs = coeff_array[:, 3]
    exponents = coeff_array[:, 4:13]  # Exponents in columns 4-12
    trig_flags = coeff_array[:, 13].astype(int)
    angle_mults = coeff_array[:, 14:17]  # Angle multipliers
    
    # Get C or S values for each term
    ccsnm_vals = np.array([float(verse) * (
        Cbar(n_val, m_val) if cs_flag == 0 else Sbar(n_val, m_val)
    ) for n_val, m_val, cs_flag in zip(n_vals, m_vals, cs_flags)])
    
    terms = coeffs * ccsnm_vals
    
    # Multiply by all power terms
    for i, power_key in enumerate(['a', 'e', 'n', 'cih', 'sih', 'eta', 'etap1', 'phi', 'RL']):
        terms *= powers[power_key] ** exponents[:, i]
    
    # Compute angles
    angles = angle_mults[:, 0] * h + angle_mults[:, 1] * lw + angle_mults[:, 2] * lf
    
    # Apply cosine or sine based on flag
    cos_terms = np.cos(angles)
    sin_terms = np.sin(angles)
    trig_terms = np.where(trig_flags == 0, cos_terms, sin_terms)
    
    # Final multiplication and sum
    return np.sum(terms * trig_terms)

def compute_all_corrections(internal_vals, external_vals=None, type_spc='mean2med', verse=1):
    """Compute all corrections"""
    akm, ecc0, n, cih, sih, eta, etap1, phi, RL, OM0, lw, lf = internal_vals
    
    # Dictionary to store corrections
    all_corrections = {key: 0.0 for key in keys}
    
    # Compute internal corrections
    for key in keys:
        resource_package = f"analcisprop.spc.internal.{type_spc}"
        resource_name = f"{key}.txt"
        
        # Use cached coefficients if available
        cache_key = (resource_package, resource_name)
        if cache_key not in _coefficient_cache:
            _coefficient_cache[cache_key] = load_spc_file(resource_package, resource_name)
        
        cf_list = _coefficient_cache[cache_key]
        all_corrections[key] += reconstruct_internal(
            cf_list, akm, ecc0, n, cih, sih, eta, etap1, phi, RL, OM0, lw, lf, verse
        )
    
    # Compute external corrections if needed
    if external_vals is not None:
        rE, xE, yE, zE, r, akm, ecc0, n, cih, sih, eta, etap1, OM0, lw, lu = external_vals
        
        for key in keys:
            resource_package = f"analcisprop.spc.external.P2.firstorder"
            resource_name = f"{key}.txt"
            
            # Use cached coefficients
            cache_key = (resource_package, resource_name)
            if cache_key not in _coefficient_cache:
                _coefficient_cache[cache_key] = load_spc_file(resource_package, resource_name)
            
            cf_list = _coefficient_cache[cache_key]
            all_corrections[key] += reconstruct_3bp(
                cf_list, verse*GME, rE, xE, yE, zE, r, akm, ecc0, n, cih, sih, eta, etap1, OM0, lw, lu
            )
    
    return all_corrections

keys = ['a', 'h', 'k', 'p', 'q', 'lM']

def mean2med(mean_eq, internal_vals, verse):
    corrections = compute_all_corrections(internal_vals, type_spc='mean2med', verse=verse)
    return [elem + corrections[key] for elem, key in zip(mean_eq, keys)]

def med2osc(med_eq, internal_vals, external_vals, verse):
    corrections = compute_all_corrections(
        internal_vals, external_vals, type_spc='med2osc', verse=verse
    )
    
    return [elem + corrections[key] for elem, key in zip(med_eq, keys)]




