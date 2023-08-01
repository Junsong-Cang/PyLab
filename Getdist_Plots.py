# ---- Triangular Plot ----
Root_1 = '/Users/cangtao/IHEPBox/Projects/GitHub/Radio_Excess_EDGES/data/multinest/Pop_II_'
Root_2 = '/Users/cangtao/IHEPBox/Projects/GitHub/Radio_Excess_EDGES/data/multinest_v2/Pop_II_'
ResultFile='/Users/cangtao/Desktop/test.png'

# ---- Initialise ----
import getdist
from getdist import plots
import os
import matplotlib.pyplot as plt

# ---- Getdist Plot ----
plt.rcParams.update({'font.family':'Times'})

samples_1 = getdist.mcsamples.loadMCSamples(Root_1)
samples_2 = getdist.mcsamples.loadMCSamples(Root_2)

p = samples_1.getParams()

g = plots.getSubplotPlotter(subplot_size = 3)
g.settings.axes_fontsize=14
g.settings.title_limit_fontsize = 14

g.triangle_plot(
    [samples_1, samples_2],
    ['fR','LX'],
    width_inch=12,
    contour_colors=['blue','red'],
    legend_labels=['EDGES','EDGES + Arcade'],
    filled = True,
    line_args=[{'lw':1.5,'ls':'-', 'color':'k'}],
    title_limit=2)

g.export(ResultFile)
print(ResultFile)

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
g.export('/Users/cangtao/Desktop/tmp.png')


# ---- Statistics ----
import getdist.plots as gplot
g = gplot.getSinglePlotter(chain_dir='/Users/cangtao/IHEPBox/Projects/GitHub/Radio_Excess_EDGES/data/multinest/')
samples = g.sampleAnalyser.samplesForRoot('Pop_II_')
print(samples.getTable(limit=1).tableTex())

os.system('open /Users/cangtao/Desktop/test.png')
