reload = 0
nm = 200
nz = 100

lm1 = 0
lm2 = 22

z1 = 0
z2 = 200
ncpu = 12
nr = 1000
swap_file = '/Users/cangtao/Desktop/tmp.npz'
TabFile = '/Users/cangtao/cloud/GitHub/HugeFiles/Halo_Profile_Table.npz'

# I cannot save files to PyLab path because of GitHub file size limit

from p21c_tools import *
from PyLab import *

m_vec = np.logspace(lm1, lm2, nm)
lzp1 = np.log10(z1+1)
lzp2 = np.log10(z2+1)
z_vec = np.logspace(lzp1, lzp2, nz) - 1

params = np.empty((2, nm*nz))

id = 0
for zid in np.arange(0, nz):
    for mid in np.arange(0, nm):
        params[0, id] = z_vec[zid]
        params[1, id] = m_vec[mid]
        id = id + 1

def model(idx):
    SaySomething()
    z = params[0,idx]
    m = params[1,idx]
    r = HaloProfile(
        z = z,
        mh = m,
        nr = nr,
        Use_Interp = False,
        map_nx = 1000,
        map_precision = 1e-4,
        mass_error = 0.05
        )
    return r

idx_vec = np.arange(0, nm*nz)

if reload:
    t1 = TimeNow()
    Tab = Parallel(n_jobs=ncpu)(delayed(model)(x) for x in idx_vec)
    Timer(t1)
    np.savez(swap_file, Tab = Tab)

Tab = np.load(swap_file)['Tab']
print(np.shape(Tab))
# z, m, r, type

ProfileTab = np.empty((nz, nm, 4, nr))

id = 0
for zid in np.arange(0, nz):
    for mid in np.arange(0, nm):
        
        ProfileTab[zid, mid, 0, :] = Tab[id, 0, :] # r_axis
        ProfileTab[zid, mid, 1, :] = Tab[id, 1, :] # RhoM
        ProfileTab[zid, mid, 2, :] = Tab[id, 2, :] # RhoC
        ProfileTab[zid, mid, 3, :] = Tab[id, 3, :] # RhoB
        id = id + 1

np.savez(TabFile, HaloProfile_Tab = ProfileTab, m = m_vec, z = z_vec, radius_size = nr)
