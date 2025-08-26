import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('agg')
import os
import emcee
import corner
import shutil

ChainFile = "data/Arcade/Arcade_chains.h5"
ConvergeFile = "data/Arcade/Arcade_status.h5"
Getdist_File_Root = 'data/Arcade/Arcade'
labels = ["$T_{\mathrm{cmb}}$", "$T_{\mathrm{R}}$", "$\gamma$"]
ParamNames = ["T_{\mathrm{cmb}}", "T_{\mathrm{R}}", "\gamma"]
ResultPath = '../Results/Arcade/'
Check_Interv = 10
Converge_Thresh = 100

# ---- Initialise ----
Triangular_Plot_File = ResultPath + 'Arcade_tri.png'
Getdist_Triangular_Plot_File = ResultPath + 'Arcade_tri.eps'
Converge_Plot_File = ResultPath + 'Arcade_Convergence.png'
ChainFile_Swap = "Chains_Swap.h5"
ConvergeFile_Swap = "Status_Swap.h5"
shutil.copyfile(ChainFile,ChainFile_Swap)
shutil.copyfile(ConvergeFile,ConvergeFile_Swap)

reader = emcee.backends.HDFBackend(ChainFile_Swap)
tau = reader.get_autocorr_time(tol=0)
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)

# Get Convergence Stats
f = h5py.File(ConvergeFile_Swap, 'r')
index = np.array(f['index'])
autocorr = np.array(f['autocorr'])
f.close()

n = Check_Interv * np.arange(1, index + 1)
y = autocorr[:index]

fig,ax=plt.subplots()
plt.plot(n, n / Converge_Thresh, "--k")
plt.plot(n, y)
plt.xlim(0, n.max())
#plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
plt.xlabel("number of steps")
plt.ylabel(r"mean $\hat{\tau}$");
fig.savefig(Converge_Plot_File,bbox_inches='tight',dpi=500)
plt.close()

# Get Triangular plot
fig = corner.corner(
    samples, 
    labels=labels,
    color='b',
    show_titles=True,
    levels=(0.95,), # for 2D
    quantiles=(0.0,0.95), # for 1D
    bins=40,
    smooth=1,
    smooth1d=1,
    )
fig.savefig(Triangular_Plot_File)

# Clean up
os.remove(ChainFile_Swap)
os.remove(ConvergeFile_Swap)
print('Triangular plot:', Triangular_Plot_File)
print('Convergence plot:', Converge_Plot_File)

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
g = plots.getSubplotPlotter()
g.triangle_plot([samples], filled = True)
g.export(Getdist_Triangular_Plot_File)
print(Getdist_Triangular_Plot_File)

