do_mcmc = 1
ndim = 3
FileRoot = '/Users/cangtao/cloud/Library/PyLab/examples/data/multinest_example/arcade_'

p1_info = {'name':'Tcmb', 'min':0, 'max':10, 'latex':'T_{\mathrm{cmb}}'}
p2_info = {'name':'TR', 'min':0, 'max':10, 'latex':'T_{\mathrm{R}}'}
p3_info = {'name':'Beta', 'min':-10, 'max':10, 'latex':'\\beta'}

import numpy as np
from pymultinest.solve import solve
import time, platform, os

'''
if platform.system() == 'Darwin':
   FileRoot = '/tmp/arcade_'
'''

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

def myprior(cube):
    '''A flat prior'''
    p1, p2, p3 = cube
    p1 = p1*(p1_info['max']-p1_info['min']) + p1_info['min']
    p2 = p2*(p2_info['max']-p2_info['min']) + p2_info['min']
    p3 = p3*(p3_info['max']-p3_info['min']) + p3_info['min']
    NewCube = [p1, p2, p3]
    return NewCube

def log_likelihood(theta):
    My_Chain_File = FileRoot + 'jsc.txt'
    Tcmb, TR, Beta = theta
    Arcade_Data = Get_Arcade_Data()
    v = Arcade_Data['Frequency_GHz']
    t0 = Arcade_Data['Temperature_K']
    dt = Arcade_Data['Uncertainty_K']
    t = model(Tcmb, TR, Beta, v)
    LnL = - np.sum((t-t0)**2/(2 * dt**2))
    
    # count calls
    cmd = 'echo ---- >> ' + My_Chain_File
    os.system(cmd)

    time.sleep(0.0) # slow things down deliberately to test speed
    return LnL

if not do_mcmc:
  from PyLab import *
  info = [p1_info, p2_info, p3_info]
  print_mcmc_info(FileRoot, info)
else:
  t1 = time.time()
  result = solve(
    LogLikelihood = log_likelihood,
    Prior=myprior, 
	n_dims=ndim,
    outputfiles_basename=FileRoot,
    resume = False,
    importance_nested_sampling = False,
    verbose=True,
    n_iter_before_update = 10)
  
  # Some post-processing, not really nessesary
  t2 = time.time()
  print('Time Used: ', t2 - t1)
  print()
  print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
  print()
  print('parameter values:')
  parameters = ['Tcmb', 'TR', 'Beta']
  for name, col in zip(parameters, result['samples'].transpose()):
	  print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
