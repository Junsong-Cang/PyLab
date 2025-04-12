import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('agg')
import os
import emcee
import corner
import shutil

Getdist_File_Root = 'data/Arcade/Arcade'
ParamNames = ["T_{\mathrm{cmb}}", "T_{\mathrm{R}}", "\gamma"]
ResultPath = '../Results/Arcade/'

# ---- Now getdist ----
import getdist
from getdist import plots, MCSamples

Param_Name_File = Getdist_File_Root+'.paramnames'
F=open(Param_Name_File,'w')
print("Tcmb         ", ParamNames[0], file=F)
print("TR           ", ParamNames[1], file=F)
print("Beta         ", ParamNames[2], file=F)
F.close()

samples = getdist.mcsamples.loadMCSamples(Getdist_File_Root)
p = samples.getParams()
g = plots.get_subplot_plotter()
g.triangular_plot([samples], filled = True)
plt.show()