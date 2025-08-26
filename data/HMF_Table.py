from p21c_tools import *
from PyLab import *

reload = 1
m1 = 1
m2 = 1e22
nm = 10000
lzp1 = np.log10(1)
lzp2 = np.log10(101)
nz = 200
swap_file = '/Users/cangtao/cloud/Library/PyLab/data/HMF_Interp_Table.npz'

z = np.logspace(lzp1, lzp2, nz) - 1
m_vec = np.logspace(np.log10(m1), np.log10(m2), nm)

def model(z_):
    m, dndm = HMF(z = z_, model = 1, Mmin = m1, Mmax = m2, nm = nm, POWER_SPECTRUM = 0, Use_Interp = False)
    r = dndm
    return r

if reload:
    t1 = TimeNow()
    dndm = Parallel(n_jobs=12)(delayed(model)(x) for x in z)
    np.savez(swap_file, dndm = dndm, z = z, m = m_vec)
    Timer(t1)

r = np.load(swap_file)
dndm = r['dndm']
print(np.shape(dndm))