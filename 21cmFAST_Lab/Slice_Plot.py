
Field_Idx = 0
#File = '/Users/cangtao/Desktop/21cmFAST-data/BoostFactor.h5'
File = '/Users/cangtao/Desktop/21cmFAST-data/BoostFactor_fiducial.h5'
#File = '/Users/cangtao/Desktop/21cmFAST-data/BoostFactor_SSCK.h5'
#File = '/Users/cangtao/Desktop/21cmFAST-data/BoostFactor_hmg.h5'
File = '/Users/cangtao/Desktop/21cmFAST-data/MCG_fid.h5'
#File = '/Users/cangtao/Desktop/21cmFAST-data/Park18.h5'
#File = '/Users/cangtao/cloud/GitHub/Radio_Excess_EDGES/tests/data/2/EoS_2021.h5'
File = '/Users/cangtao/Desktop/21cmFAST-data/BoostFactor/main.h5'
# File = '/Users/cangtao/Desktop/21cmFAST-data/BoostFactor/Fiducial.h5'
FileName = '/Users/cangtao/Desktop/21cmFAST-data/tmp/tmp.h5'

LineWidth = 2
FontSize = 20

import matplotlib.pyplot as plt
import matplotlib, h5py
import numpy as np
from matplotlib.colors import LogNorm
import py21cmfast as p21c
from cosmo_tools import *

Fields = [
    [0, 'brightness_temp', 'global_brightness_temp', '$T_{21}\ [{\mathrm{mK}}]$', 'linear'],
    [1, 'Ts_box', 'global_Ts', '$T_{\mathrm{s}}\ [{\mathrm{K}}]$', 'log'],
    [2, 'Tk_box', 'global_Tk', '$T_{\mathrm{k}}\ [{\mathrm{K}}]$', 'log'],
    [3, 'xH_box', 'global_xH', '$x_{\mathrm{H}}$', 'linear'],
    [4, 'Trad_box', 'global_Trad', '$T_{\mathrm{radio}}\ [{\mathrm{K}}]$', 'log'],
    [5, 'Boost_box', 'global_Boost', '$B$', 'log'],
    [6, 'xe_box', 'global_xe', '$x_{\mathrm{e}}$', 'log'],
    [7, 'SFRD_box', 'global_SFRD', 'SFRD', 'log'],
    [8, 'SFRD_MINI_box', 'global_SFRD_MINI', 'SFRD III', 'log'],
    ]

lc = p21c.LightCone.read(File)
f = h5py.File(File, 'r')
try:
    z = lc.lightcone_redshifts
except:
    z = get_lc_redshifts(lc)

BOX_LEN = lc.user_params.BOX_LEN
HII_DIM = lc.user_params.HII_DIM
x = np.linspace(0, BOX_LEN, HII_DIM)

if Field_Idx == 6:
    cmd = 'y3d = lc.' + Fields[3][1]
    cmd2 = 'global_y = lc.' + Fields[3][2]
    exec(cmd)
    exec(cmd2)
    y3d = 1 - y3d
    global_y = 1 - global_y

else:
    cmd = 'y3d = lc.' + Fields[Field_Idx][1]
    cmd2 = 'global_y = lc.' + Fields[Field_Idx][2]
    exec(cmd)
    exec(cmd2)

y = y3d[1,:,:]
EoR_colour = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap',\
             [(0, 'white'),(0.33, 'yellow'),(0.5, 'orange'),(0.68, 'red'),\
              (0.83333, 'black'),(0.9, 'blue'),(1, 'cyan')])
plt.register_cmap(cmap=EoR_colour)

# Set font
plt.rcParams.update({'font.family':'Times'})
plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots()
fig.set_size_inches(10, 1.8)

z, x = np.meshgrid(z, x)
if Fields[Field_Idx][4] == 'linear':
    # c=ax.pcolor(z, x, y,cmap='jet')
    c=ax.pcolor(z, x, y,cmap = EoR_colour, vmin = -140, vmax = 30)
else:
    c=ax.pcolor(z, x, y,cmap='jet',norm = LogNorm(vmin=global_y.min(), vmax=global_y.max()))

plt.xlabel('$z$',fontsize=FontSize,fontname='Times New Roman')
plt.ylabel('Mpc',fontsize=FontSize,fontname='Times New Roman')
plt.xticks(size=FontSize)
plt.yticks(np.linspace(0, BOX_LEN, 4), size=FontSize)
plt.xscale('log')
clb = plt.colorbar(c)
plt.title(Fields[Field_Idx][3],fontsize=FontSize)

plt.tight_layout()

plt.savefig('/Users/cangtao/Desktop/tmp.png',dpi=1000)
print(HII_DIM)
print(BOX_LEN)
plt.show()
