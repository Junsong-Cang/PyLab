
Field_Idx = 0
File = '/Users/cangtao/Desktop/21cmFAST-data/EOS_2021.h5'

LineWidth = 2
FontSize = 15

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import py21cmfast as p21c

Fields = [
    [0, 'brightness_temp', 'global_brightness_temp', '$T_{21}\ [{\mathrm{mK}}]$', 'linear'],
    [1, 'Ts_box', 'global_Ts', '$T_{\mathrm{s}}\ [{\mathrm{K}}]$', 'log'],
    [2, 'Tk_box', 'global_Tk', '$T_{\mathrm{k}}\ [{\mathrm{K}}]$', 'log'],
    [3, 'xH_box', 'global_xH', '$x_{\mathrm{H}}$', 'linear'],
    [4, 'Trad_box', 'global_Trad', '$T_{\mathrm{radio}}\ [{\mathrm{K}}]$', 'log'],
]

lc = p21c.LightCone.read(File)

z = lc.lightcone_redshifts
BOX_LEN = lc.user_params.BOX_LEN
HII_DIM = lc.user_params.HII_DIM
x = np.linspace(0, BOX_LEN, HII_DIM)

cmd = 'y3d = lc.' + Fields[Field_Idx][1]
cmd2 = 'global_y = lc.' + Fields[Field_Idx][2]
exec(cmd)
exec(cmd2)

y = y3d[1,:,:]

# Set font
plt.rcParams.update({'font.family':'Times'})
plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots()
fig.set_size_inches(10, 3)

z, x = np.meshgrid(z, x)
if Fields[Field_Idx][4] == 'linear':
    c=ax.pcolor(z, x, y,cmap='jet')
else:
    c=ax.pcolor(z, x, y,cmap='jet',norm = LogNorm(vmin=global_y.min(), vmax=global_y.max()))

plt.xlabel('$z$',fontsize=FontSize,fontname='Times New Roman')
plt.ylabel('Mpc',fontsize=FontSize,fontname='Times New Roman')
plt.xticks(size=FontSize)
plt.yticks(size=FontSize)
clb = plt.colorbar(c)
plt.title(Fields[Field_Idx][3],fontsize=FontSize)

plt.tight_layout()

plt.savefig('/Users/cangtao/Desktop/tmp.png',dpi=1000)
