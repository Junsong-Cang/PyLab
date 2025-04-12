do_mcmc = 0
ndim = 3
FileRoot = 'data/arcade_'

p1_info = {'name':'Tcmb', 'min':0, 'max':10, 'latex':'T_{\mathrm{cmb}}'}
p2_info = {'name':'TR', 'min':0, 'max':10, 'latex':'T_{\mathrm{R}}'}
p3_info = {'name':'Beta', 'min':-10, 'max':10, 'latex':'\\beta'}

import numpy as np
from pymultinest.solve import solve
import PyLab as PL
import time

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

def myprior(cube):
	'''A flat prior'''
	p1, p2, p3 = cube
	p1 = p1*(p1_info['max']-p1_info['min']) + p1_info['min']
	p2 = p2*(p2_info['max']-p2_info['min']) + p2_info['min']
	p3 = p3*(p3_info['max']-p3_info['min']) + p3_info['min']
	NewCube = [p1, p2, p3]
	return NewCube

def log_likelihood(theta):
  Tcmb, TR, Beta = theta
  t0 = Arcade_Data['Temperature_K']
  dt = Arcade_Data['Uncertainty_K']
  t = model(Tcmb, TR, Beta)
  Chi2 = np.sum(((t-t0)/dt)**2)
  LnL = - Chi2/2
  PL.SaySomething()
  # time.sleep(0.0)
  return LnL

if do_mcmc:
  t1 = time.time()
  result = solve(
      LogLikelihood = log_likelihood,
      Prior=myprior, 
	    n_dims=ndim,
      outputfiles_basename=FileRoot,
      n_live_points=100 * ndim,
      resume = False,
      verbose = True,
      n_iter_before_update = 10)
  # Some post-processing, not really nessesary
  t2 = time.time()
  print('Time Used: ', t2 - t1)
  print()
  # print(result['logZ'])
  # print(result['logZerr'])
  print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
  print()
  print('parameter values:')
  parameters = ['Tcmb', 'TR', 'Beta']
  for name, col in zip(parameters, result['samples'].transpose()):
	  print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
else:
  info = [p1_info, p2_info, p3_info]
  PL.print_mcmc_info(FileRoot, info)
  z = np.linspace(4, 30, 40)
  def T_Arcade(theta):
    '''
    Arcade excess level (@v21) at z, outputs in K, see 1802.07432
    '''
    Tcmb, TR, Beta = theta
    v21 = 1.429
    v0 = v21/(1+z)
    t0 = TR * (v0**Beta) # arcade model
    t = (1+z) * t0
    return t
  current_directory = PL.os.getcwd()
  FileRoot = current_directory+'/'+FileRoot
  TA = PL.mcmc_derived_stat(
      model_function = T_Arcade,
      FileRoot = FileRoot,
      NewRoot = FileRoot + 'tmp',
      print_status = 1,
      ncpu = 12,
      remove_NaN = 1)
  BestFit = PL.Read_MultiNest_BestFit(Root=FileRoot)
  T_Best = T_Arcade(BestFit)
  np.savez(FileRoot+'T_excess_posterior.npz', TA = TA, z = z, T_Best = T_Best)
  print(current_directory)
