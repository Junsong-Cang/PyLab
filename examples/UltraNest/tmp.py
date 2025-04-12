Root_1 = '/Users/cangtao/FileVault/Soft/PyLab/examples/UltraNest/data/chains/weighted_post'
Root_2 = '/Users/cangtao/FileVault/Soft/PyLab/examples/MultiNest/data/arcade_'
ResultFile='/Users/cangtao/Desktop/tmp.pdf'

import PyLab as PL
from getdist import plots
import os
import matplotlib.pyplot as plt
import getdist

'''
FileRoot = '/Users/cangtao/FileVault/Soft/PyLab/examples/UltraNest/data/chains/weighted_post'
p1_info = {'name':'Tcmb', 'min':0, 'max':10, 'latex':'T_{\mathrm{cmb}}'}
p2_info = {'name':'TR', 'min':0, 'max':10, 'latex':'T_{\mathrm{R}}'}
p3_info = {'name':'Beta', 'min':-10, 'max':10, 'latex':'\\beta'}
info = [p1_info, p2_info, p3_info]
PL.print_mcmc_info(FileRoot, info)
'''
plt.rcParams.update({'font.family':'Times'})
plt.rcParams['text.usetex'] = True

samples_1 = getdist.mcsamples.loadMCSamples(Root_1)
samples_2 = getdist.mcsamples.loadMCSamples(Root_2)

p = samples_1.getParams()

g = plots.getSubplotPlotter(subplot_size = 3)
g.settings.axes_fontsize=14
g.settings.title_limit_fontsize = 12
g.settings.lab_fontsize =14
g.settings.axes_labelsize = 14 # Size of axis label
g.settings.legend_fontsize = 14 # Legend size

g.triangle_plot(
    [samples_1, samples_2],
    width_inch=12,
    contour_colors=['blue','red'],
    legend_labels=['UN','MN'],
    filled = True,
    line_args=[{'lw':1.5,'ls':'-', 'color':'b'},
               {'lw':1.5,'ls':'-', 'color':'r'}],
    title_limit = 2)

plt.savefig(ResultFile)
