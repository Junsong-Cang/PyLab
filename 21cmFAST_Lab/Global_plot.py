
Field_Idx = 0

File = '/Users/cangtao/Desktop/21cmFAST-data/BoostFactor.h5'
#File = '/Users/cangtao/Desktop/21cmFAST-data/BoostFactor_fiducial.h5'
#File = '/Users/cangtao/Desktop/21cmFAST-data/BoostFactor_SSCK.h5'
File = '/Users/cangtao/Desktop/21cmFAST-data/BoostFactor_hmg.h5'
File = '/Users/cangtao/Desktop/21cmFAST-data/EOS_2021_no_mini.h5'
File = '/Users/cangtao/Desktop/21cmFAST-data/tmp_EOS_2021.h5'
File = '/Users/cangtao/Desktop/21cmFAST-data/MCG_fid.h5'
File = '/Users/cangtao/Desktop/21cmFAST-data/tmp.h5'

File = '/Users/cangtao/Desktop/21cmFAST-data/Boost/fiducial.h5'
File = '/Users/cangtao/Desktop/21cmFAST-data/Boost/boost_less.h5'
File = '/Users/cangtao/Desktop/21cmFAST-data/Boost/hmg_boost.h5'
File = '/Users/cangtao/Desktop/21cmFAST-data/Boost/main.h5'

File = '/Users/cangtao/Desktop/21cmFAST-data/MCG_Plots/MCG_0.h5'

LineWidth = 3
FontSize = 25

import py21cmfast as p21c
import matplotlib.pyplot as plt
import numpy as np
from cosmo_tools import *
# Format : [index, lc_name, global_name, LaTex, yscale]
Fields = [
    [0, 'brightness_temp', 'global_brightness_temp', '$T_{21}\ [{\mathrm{mK}}]$', 'linear'],
    [1, 'Ts_box', 'global_Ts', '$T_{\mathrm{s}}\ [{\mathrm{K}}]$', 'log'],
    [2, 'Tk_box', 'global_Tk', '$T_{\mathrm{k}}\ [{\mathrm{K}}]$', 'log'],
    [3, 'xH_box', 'global_xH', '$x_{\mathrm{H}}$', 'linear'],
    [4, 'Trad_box', 'global_Trad', '$T_{\mathrm{radio}}\ [{\mathrm{K}}]$', 'log'],
    [5, 'Boost_box', 'global_Boost', '$B$', 'log'],
]

lc = p21c.LightCone.read(File)
z = lc.node_redshifts
cmd = 'x = lc.' + Fields[Field_Idx][2]
exec(cmd)

plt.rcParams.update({'font.family':'Times'})
plt.rcParams['text.usetex'] = True

plt.plot(z, x, 'k', linewidth=LineWidth)
plt.yscale(Fields[Field_Idx][4])
plt.xlabel('$z$',fontsize=FontSize,fontname='Times New Roman')
plt.ylabel(Fields[Field_Idx][3],fontsize=FontSize,fontname='Times New Roman')

plt.xticks(size=FontSize)
plt.yticks(size=FontSize)
plt.tight_layout()

plt.savefig('/Users/cangtao/Desktop/tmp.png', dpi=1000)
t = lc2tau(File)
print(t)
