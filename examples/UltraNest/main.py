FileRoot = 'data/'
Vectorize = True
p1_info = {'name':'Tcmb', 'min':0, 'max':10, 'latex':'T_{\mathrm{cmb}}'}
p2_info = {'name':'TR', 'min':0, 'max':10, 'latex':'T_{\mathrm{R}}'}
p3_info = {'name':'Beta', 'min':-10, 'max':10, 'latex':'\\beta'}

import numpy as np
import PyLab as PL
import time, ultranest, os

infos = [p1_info, p2_info, p3_info]
def Get_Arcade_Data():
  # Radio tempertures from Tab.4 of ARCADE2 paper (0901.0555), beware of the differences of antenna temp and thermo temp
  Arcade_Data = {
    'Frequency_GHz': np.array([0.022, 0.045, 0.408, 1.42, 3.2, 3.41, 7.97, 8.33, 9.72, 10.49, 29.5, 31, 90]),
    'Temperature_K': np.array([21200, 4355, 16.24, 3.213, 2.792, 2.771, 2.765, 2.741, 2.732, 2.732, 2.529, 2.573, 2.706]),
    'Uncertainty_K': np.array([5125, 520, 3.4, 0.53, 0.01, 0.009, 0.014, 0.016, 0.006, 0.006, 0.155, 0.076, 0.019])}
  return Arcade_Data
Arcade_Data = Get_Arcade_Data()

def model(Tcmb, TR, Beta):
  v = Arcade_Data['Frequency_GHz']
  v0 = 1.0
  T = Tcmb + TR*(v/v0)**Beta
  return T

def FlatPrior_Kernel(cube):
	'''A flat prior'''
	p1, p2, p3 = cube
	p1 = p1*(p1_info['max']-p1_info['min']) + p1_info['min']
	p2 = p2*(p2_info['max']-p2_info['min']) + p2_info['min']
	p3 = p3*(p3_info['max']-p3_info['min']) + p3_info['min']
	NewCube = [p1, p2, p3]
	return NewCube

def LogLike_Kernel(theta):
  Tcmb, TR, Beta = theta
  t0 = Arcade_Data['Temperature_K']
  dt = Arcade_Data['Uncertainty_K']
  t = model(Tcmb, TR, Beta)
  Chi2 = np.sum(((t-t0)/dt)**2)
  LnL = - Chi2/2
  PL.SaySomething()
  return LnL

def FlatPrior(thetas):
    param_shape = np.shape(thetas)
    param_len = param_shape[0]
    new_theta = []
    for idx in np.arange(0, param_len):
        theta = thetas[idx][:]
        new_theta.append(FlatPrior_Kernel(theta))
    new_theta = np.array(new_theta)
    return new_theta
    
def LogLike(thetas):
    param_shape = np.shape(thetas)
    param_len = param_shape[0]
    LnL = []
    for idx in np.arange(0, param_len):
        theta = thetas[idx][:]
        LnL.append(LogLike_Kernel(theta))
    LnL = np.array(LnL)
    return LnL

if Vectorize:
    my_likelihood = LogLike
    Prior = FlatPrior
else:
    my_likelihood = LogLike_Kernel
    Prior = FlatPrior_Kernel
t1 = time.time()

param_names = [p1_info['name'], p2_info['name'], p3_info['name']]
sampler = ultranest.ReactiveNestedSampler(
    param_names = param_names, 
    loglike = my_likelihood, 
    transform = Prior,
    log_dir = FileRoot, 
    resume = True,
    vectorized = Vectorize,
    ndraw_min = 1000) # Number of param vectors
result = sampler.run(min_num_live_points=100*len(infos))

t2 = time.time()
'''
LnZ_DataFile = FileRoot+'Bayesian_Evidence.dat'
str1 = 'LnZ = '+"{:.4f}".format(result['logz'])
str2 = 'LnZ_Err = '+"{:.4f}".format(result['logzerr'])
str3 = 'Time used = '+"{:.4f}".format(t2 - t1)
cmd1 = 'echo ' + str1 + '>>' +LnZ_DataFile
cmd2 = 'echo ' + str2 + '>>' +LnZ_DataFile
cmd3 = 'echo ' + str3 + '>>' +LnZ_DataFile
os.system(cmd1)
os.system(cmd2)
os.system(cmd3)
'''
PL.Print_UntraFast_info(result=result, LogPath=FileRoot)
sampler.print_results()
log_z = result['logz']
print("Bayesian evidence (log Z): ", log_z)
log_z_error = result['logzerr']
print("Uncertainty on Bayesian evidence (log Z): ", log_z_error)
print('Time used:', t2 - t1,', Vectorized =', Vectorize)
