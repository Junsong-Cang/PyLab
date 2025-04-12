
SlowFactor = 10
FileRoot = 'chains/arcade_'
ParamPrior = {
  'Tcmb': {'min':0, 'max':10, 'fiducial':2.728},
  'TR': {'min':0, 'max':10, 'fiducial':1.19},
  'Beta': {'min':-10, 'max':10, 'fiducial':-2.62}
}
parameters = ['Tcmb', 'TR', 'Beta']
ParamNames = ['T_{\mathrm{cmb}}', 'T_{\mathrm{R}}', '\\beta']

# ---- Initialising ----
import numpy as np
import pymultinest, shutil, os, time
start_time = time.time()

def Print_paramname(Params, LaTex, FileRoot):
	FileName = FileRoot+'.paramnames'
	FileName_jsc = FileRoot + 'jsc' + '.paramnames'
	F=open(FileName,'w')
	for idx in np.arange(0,len(Params)):
		print(Params[idx],'		',LaTex[idx], file = F)
	F.close()
	shutil.copyfile(FileName,FileName_jsc)

def Get_Arcade_Data():
  # Radio tempertures from Tab.4 of ARCADE2 paper (0901.0555), beware of the differences of antenna temp and thermo temp
  Arcade_Data = {
    'Frequency_GHz': np.array([0.022, 0.045, 0.408, 1.42, 3.2, 3.41, 7.97, 8.33, 9.72, 10.49, 29.5, 31, 90]),
    'Temperature_K': np.array([21200, 4355, 16.24, 3.213, 2.792, 2.771, 2.765, 2.741, 2.732, 2.732, 2.529, 2.573, 2.706]),
    'Uncertainty_K': np.array([5125, 520, 3.4, 0.53, 0.01, 0.009, 0.014, 0.016, 0.006, 0.006, 0.155, 0.076, 0.019])
    }
  return Arcade_Data

def model(Tcmb, TR, Beta, v):
  v0 = 1.0
  T = Tcmb + TR*(v/v0)**Beta
  return T

def myprior(cube, ndim, nparams):
	global ParamPrior
	p1_range = ParamPrior['Tcmb']
	p2_range = ParamPrior['TR']
	p3_range = ParamPrior['Beta']
	cube[0] = cube[0]*(p1_range['max']-p1_range['min']) + p1_range['min']
	cube[1] = cube[1]*(p2_range['max']-p2_range['min']) + p2_range['min']
	cube[2] = cube[2]*(p3_range['max']-p3_range['min']) + p3_range['min']

def log_likelihood(cube, ndim, nparams):
	Tcmb = cube[0]
	TR = cube[1]
	Beta = cube[2]
	Arcade_Data = Get_Arcade_Data()
	v = Arcade_Data['Frequency_GHz']
	t0 = Arcade_Data['Temperature_K']
	dt = Arcade_Data['Uncertainty_K']
	t = model(Tcmb, TR, Beta, v)
	LnL = - np.sum((t-t0)**2/(2 * dt**2))
	return LnL

n_params = len(parameters)

result = pymultinest.run(
	LogLikelihood = log_likelihood, 
	Prior = myprior,
	n_dims = n_params,
	n_params = n_params,
	outputfiles_basename = FileRoot,
	resume = False, 
	n_live_points=100 * n_params,
	verbose = True)

Print_paramname(parameters, ParamNames, FileRoot)
TimeUsed = time.time()-start_time
print('---- Time used ----')
print(TimeUsed)
