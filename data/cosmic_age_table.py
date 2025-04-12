from p21c_tools import *

reload = 1
nz = 10000
zp = np.logspace(0, 10, nz)
z = zp-1

LgZp_axis = np.log10(zp)
LgT_axis = np.zeros(nz)

if reload:
    t1 = TimeNow()
    r = np.zeros(nz)
    for idx in np.arange(0, nz):
        z_ = z[idx]
        r[idx] = z2t(z = z_, nz = 10000, Use_interp = 0)
        print(idx/nz)
    
    LgT_axis = np.log10(r)
    np.savez('cosmic_age_table.npz', LgZp_axis = LgZp_axis, LgT_axis = LgT_axis)
    Timer(t1)

plt.loglog(zp, r)
plt.show()
