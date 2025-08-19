Root='/Users/cangtao/FileVault/Projects/Radio_Excess_EDGES/data/280_FG7_Tcal_LogST_HiRes/280_FG7_Tcal_LogST_HiRes_'

import getdist
from getdist import plots
import os
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'font.family':'Times',
    'text.latex.preamble': r'\usepackage{newtxtext,newtxmath}'})

samples = getdist.mcsamples.loadMCSamples(Root)
p = samples.getParams()

g = plots.getSubplotPlotter(subplot_size = 3)
g.settings.axes_fontsize=14
g.settings.title_limit_fontsize = 12
g.settings.lab_fontsize =14
g.settings.axes_labelsize = 14 # Size of axis label
g.settings.legend_fontsize = 14 # Legend size
g.triangle_plot(
    samples, 
    # ['fR','LX', 'zcut'], # select params, default uses all
    width_inch=12,
    contour_colors=['blue'],
    legend_labels=['EDGES+ARCADE+Planck'],
    filled = True,
    line_args=[{'lw':1.5,'ls':'-', 'color':'k'}],
    title_limit = 2,
    #param_limits = {'fR': [4.5, 5.5], 'LX' : [39, 42], 'zcut' : [14, 20]}, # set axis limits
    # markers = {'fR' : fR_best,
    #           'LX' : LX_best}, # Mark value, use best-fit for this example
    marker_args = {'lw' : 1.5}, # Marker setting, lw - LineWidth, can also set color etc
    )

# Don't recommend g.export but that has a lower resolution
plt.tight_layout()
plt.savefig('/Users/cangtao/Desktop/tmp.pdf')