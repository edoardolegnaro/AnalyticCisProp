import heyoka as hk
from analcisprop.constants import AxV, BxV, wxV, AyV, ByV, wyV, AzV, BzV, wzV, GME, GML, lambda_moon
import time
import os
import numpy as np

a, h, k, p, q, lM = hk.make_vars("a", "h", "k", "p", "q", "lM")
t = hk.time
OM = hk.atan2(q, p)
lw = hk.atan2(k, h)
ecc = hk.sqrt(h**2 + k**2)
eta = hk.sqrt(1 - ecc)
sih = hk.sqrt(q**2 + p**2)
cih = hk.sqrt(1 - sih**2)
n = hk.sqrt(GML / a**3)

xE = 382469.63+ np.sum([AxV[j] * hk.cos(wxV[j] * t) + BxV[j] * hk.sin(wxV[j] * t) for j in range(len(AxV))], axis=0)
yE = np.sum([AyV[j] * hk.cos(wyV[j] * t) + ByV[j] * hk.sin(wyV[j] * t) for j in range(len(AyV))], axis=0)
zE = np.sum([AzV[j] * hk.cos(wzV[j] * t) + BzV[j] * hk.sin(wzV[j] * t) for j in range(len(AzV))], axis=0)

rE = hk.sqrt( xE**2 + yE**2 + zE**2 )

current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)
variables = ['hdot', 'kdot', 'lMdot', 'pdot', 'qdot']
VF = {}
for var in variables:
       with open(os.path.join(current_dir_path, 'secular_equations', 'heyoka_equations', f'{var}.txt'), "r") as f:
              VF[var] = eval(f.read())

ic0_eq = [2.03775416e+03, 9.52627774e-02, 2.95478462e-02, 9.98273644e-02,8.40936532e-02, 1.25663046e+01]

print('Compiling the Double Averaged Taylor integrator ... (this is done only once)')

start_time = time.time()

ta = hk.taylor_adaptive(
    sys = [(a, a*0.0), (h, VF['hdot']), (k, VF['kdot']), 
           (p, VF['pdot']), (q, VF['qdot']), (lM, VF['lMdot']+ n-lambda_moon)],
    state = ic0_eq,
    tol=1e-9,
    compact_mode = True
)

end_time = time.time()

print('Done, in')
print("--- %s seconds ---" % (end_time - start_time))

print("\nHeyoka Taylor integrator:\n", ta)

print("\nTaylor order:\n", len(ta.decomposition) - 12)

print("\n")

def propagate(tmax, dt, ic):
    ta.time = 0
    ta.state[:] = ic
    tgrid = np.linspace(0.0, tmax, int(1 + tmax/dt), endpoint = True)
    out = ta.propagate_grid(tgrid)
    return out[5]