Root = '/Users/cangtao/FileVault/Projects/SKA_DC3/DataSet/PseudoLikelihood/data/PseudoLikelihood_'

# ---- Initialise ----
import getdist
from getdist import plots
import os
import matplotlib.pyplot as plt

# ---- Getdist Plot ----
plt.rcParams.update({
    'text.usetex': True,
    'font.family':'Times',
    'text.latex.preamble': r'\usepackage{newtxtext,newtxmath}'})

samples_1 = getdist.mcsamples.loadMCSamples(Root)
p = samples_1.getParams()

g = plots.getSubplotPlotter(subplot_size = 3)
g.settings.axes_fontsize=14
g.settings.title_limit_fontsize = 12
g.settings.lab_fontsize =14
g.settings.axes_labelsize = 14 # Size of axis label
g.settings.legend_fontsize = 14 # Legend size

g.triangle_plot(
    samples_1,
    width_inch=12,
    contour_colors=['blue'],
    # legend_labels=['EDGES','EDGES + Arcade'],
    filled = True,
    line_args=[{'lw':1.5,'ls':'-', 'color':'k'}],
    title_limit = 2,
    )

plt.tight_layout()
plt.savefig('/Users/cangtao/Desktop/tmp.pdf')
