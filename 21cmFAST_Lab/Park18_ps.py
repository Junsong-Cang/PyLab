
reload = 0
z_ = 8.6
k_ = 0.15
LineWidth = 2
FontSize = 18

lc_file = '/Users/cangtao/Desktop/21cmFAST-data/Park18.h5'
ps_file = '/Users/cangtao/Desktop/21cmFAST-data/Park18.npz'
ps_file = '/Users/cangtao/Desktop/21cmFAST-data/EOS_2021.npz'

from p21c_tools import *

if reload:
    p21c_ps_kernel(LC_file = lc_file, npz_file = ps_file, nk = 50, nz = 50)

k, psk = read_psk(file = ps_file, z = z_)
z, psz = read_psz(file = ps_file, k = k_)

plt.rcParams.update({'font.family':'Times'})
plt.rcParams['text.usetex'] = True

fig, axs = plt.subplots(1, 2, sharex = False, sharey = False)
fig.set_size_inches(8, 4)

axs[0].semilogy(z, psz, 'k', linewidth = LineWidth)
axs[0].set_title('$k = 0.1$',fontsize=FontSize)
axs[0].set_xlabel('$z$',fontsize=FontSize,fontname='Times New Roman')
axs[0].set_ylabel('$\Delta^2_{21}\ [{\mathrm{mK^2}}]$',fontsize=FontSize,fontname='Times New Roman')
axs[0].tick_params(axis='both', which='both', labelsize = FontSize)
#axs[0].set_xlim(-3.2, 3.2)
#axs[0].set_ylim(-1.2, 1.2)

axs[1].loglog(k, psk, 'k', linewidth = LineWidth)
axs[1].set_title('$z = 12.2$',fontsize=FontSize)
axs[1].set_xlabel('$k\ [{\mathrm{Mpc^{-1}}}]$',fontsize=FontSize,fontname='Times New Roman')
axs[1].set_ylabel('$\Delta^2_{21}\ [{\mathrm{mK^2}}]$',fontsize=FontSize,fontname='Times New Roman')
#axs[1].set_xlim(-3.2, 3.2)
#axs[1].set_ylim(0, 1.2)

plt.xticks(size=FontSize)
plt.yticks(size=FontSize)

plt.tight_layout()
plt.savefig('/Users/cangtao/Desktop/tmp.png', dpi=1000)
