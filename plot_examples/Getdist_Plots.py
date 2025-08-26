# ---- Triangular Plot ----
Root_1 = '/Users/cangtao/cloud/GitHub/Radio_Excess_EDGES/data/1_zcut/1_zcut_'
Root_2 = '/Users/cangtao/cloud/GitHub/Radio_Excess_EDGES/data/15_zcut_Lite_Tunned/15_zcut_Lite_Tunned_'
ResultFile='/Users/cangtao/Desktop/tmp.pdf'

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

samples_1 = getdist.mcsamples.loadMCSamples(Root_1)
samples_2 = getdist.mcsamples.loadMCSamples(Root_2)

# Best-Fit, in the order of chain
max_likelihood_sample = samples_1.samples[samples_1.loglikes.argmax()]
fR_best = max_likelihood_sample[0] # determined by chain order
LX_best = max_likelihood_sample[1]

p = samples_1.getParams()

g = plots.getSubplotPlotter(subplot_size = 3)
g.settings.axes_fontsize=14
g.settings.title_limit_fontsize = 12
g.settings.lab_fontsize =14
g.settings.axes_labelsize = 14 # Size of axis label
g.settings.legend_fontsize = 14 # Legend size

g.triangle_plot(
    [samples_1, samples_2],
    ['fR','LX', 'zcut'], # select params, default uses all
    width_inch=12,
    contour_colors=['blue','red'],
    legend_labels=['EDGES','EDGES + Arcade'],
    filled = True,
    line_args=[{'lw':1.5,'ls':'-', 'color':'k'}],
    title_limit = 2,
    param_limits = {'fR': [4.5, 5.5], 'LX' : [39, 42], 'zcut' : [14, 20]}, # set axis limits
    markers = {'fR' : fR_best,
               'LX' : LX_best}, # Mark value, use best-fit for this example
    marker_args = {'lw' : 1.5}, # Marker setting, lw - LineWidth, can also set color etc
    )

# Don't recommend g.export but that has a lower resolution
plt.tight_layout()
plt.savefig('/Users/cangtao/Desktop/tmp.pdf')
#g.export(ResultFile)
#print(ResultFile)

# ---- 2D ----
import getdist
from getdist import plots
import matplotlib.pyplot as plt

Root = '/Users/cangtao/cloud/GitHub/Radio_Excess_EDGES/Example_pymultinest/chains/arcade_'

plt.rcParams.update({'font.family':'Times'})
samples = getdist.mcsamples.loadMCSamples(Root)

g = plots.get_single_plotter(width_inch = 4)
samples.updateSettings({'contours': [0.68, 0.95]})
g.settings.num_plot_contours = 2
g.plot_2d(samples, 'Tcmb', 'TR', filled = True)
# g.add_legend(['sim 1', 'sim 2'], colored_text=True)
g.export('/Users/cangtao/Desktop/tmp_2.png')


# ---- Statistics ----
import getdist.plots as gplot
g = gplot.getSinglePlotter(chain_dir='/Users/cangtao/cloud/GitHub/Radio_Excess_EDGES/Example_pymultinest/chains/')
samples = g.sampleAnalyser.samplesForRoot('arcade_')
print(samples.getTable(limit=1).tableTex())
s = samples.getLikeStats()

os.system('open /Users/cangtao/Desktop/tmp_3.png')

# ---- maximum likelihood ----

import getdist.plots as gdplt
import getdist.mcsamples as mcsamples

# Load the MCMC samples
samples = mcsamples.loadMCSamples('/Users/cangtao/cloud/GitHub/Radio_Excess_EDGES/Example_pymultinest/chains/arcade_')

# Calculate the maximum likelihood value of a parameter
max_likelihood_sample = samples.samples[samples.loglikes.argmax()]
# This gets the sample with the highest likelihood

# Print the parameter values at maximum likelihood
print("Maximum Likelihood Parameter Values:")
for i, param_name in enumerate(samples.paramNames.names):
    print(f"{param_name} : {max_likelihood_sample[i]}")
