import getdist
from getdist import plots, MCSamples

FileRoot = '/Users/cangtao/cloud/GitHub/Radio_Excess_EDGES/Example_pymultinest/chains/arcade_'
PlotFile = '/Users/cangtao/cloud/GitHub/Radio_Excess_EDGES/Results/Arcade/arcade_tri.eps'

samples = getdist.mcsamples.loadMCSamples(FileRoot)
p = samples.getParams()
g = plots.getSubplotPlotter()
g.triangle_plot(
    samples,
    width_inch=12,
    contour_colors=['blue'],
    filled = True,
    line_args=[{'lw':1.5,'ls':'-', 'color':'k'}],
    title_limit=2)

g.export(PlotFile)

print(PlotFile)

