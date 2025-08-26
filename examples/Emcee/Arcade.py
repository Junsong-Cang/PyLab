import emcee
import numpy as np
import os
import h5py
import shutil
import os

nwalkers = 6
n_samples = 100000
DataPath = 'data/Arcade/'
Prefix = 'Arcade'

Check_Interv = 10
# Convergence Ctiteria
Converge_Thresh = 100

ParamPrior = {
  'Tcmb': {'min':0, 'max':10, 'fiducial':2.728},
  'TR': {'min':0, 'max':10, 'fiducial':1.19},
  'Beta': {'min':-10, 'max':10, 'fiducial':-2.62}
}

# ---- Initialise ----
ndim = len(ParamPrior)
File_Root = DataPath + Prefix
ChainFile = File_Root + '_chains.h5'
ConvergeFile = File_Root + '_status.h5'
Getdist_ChainFile = File_Root+'.txt'
try:
    shutil.rmtree(DataPath)
except FileNotFoundError:
	pass
os.mkdir(DataPath)

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

def log_likelihood(theta):
  Tcmb, TR, Beta = theta
  Arcade_Data = Get_Arcade_Data()
  v = Arcade_Data['Frequency_GHz']
  t0 = Arcade_Data['Temperature_K']
  dt = Arcade_Data['Uncertainty_K']
  t = model(Tcmb, TR, Beta, v)
  LnL = - (t-t0)**2/(2 * dt**2)
  return np.sum(LnL)

def log_prior(theta):
  global ParamPrior
  Tcmb, TR, Beta = theta
  Tcmb_prior = ParamPrior['Tcmb']
  TR_prior = ParamPrior['TR']
  Beta_prior = ParamPrior['Beta']
  Tcmb_OK = Tcmb_prior['min'] < Tcmb < Tcmb_prior['max']
  TR_OK = TR_prior['min'] < TR < TR_prior['max']
  Beta_OK = Beta_prior['min'] < Beta < Beta_prior['max']
  if Tcmb_OK and TR_OK and Beta_OK:
    return 0
  else:
    return -np.inf

def log_probability(theta):
  global Getdist_ChainFile
  LogP = log_prior(theta) + log_likelihood(theta)
  F=open(Getdist_ChainFile,'a')
  Wight = 1.0
  p1, p2, p3 = theta
  Chi2 = -2 * LogP
  print("{0:.5E}".format(Wight), "    {0:.5E}".format(Chi2), "    {0:.5E}".format(p1), "    {0:.5E}".format(p2), "    {0:.5E}".format(p3), file=F)
  F.close()
  return LogP

p1_start = ParamPrior['Tcmb']['fiducial']
p2_start = ParamPrior['TR']['fiducial']
p3_start = ParamPrior['Beta']['fiducial']

Start_Location=[p1_start, p2_start, p3_start] + 1e-4 * np.random.randn(nwalkers, ndim)

backend = emcee.backends.HDFBackend(ChainFile)
backend.reset(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,backend=backend)

# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(n_samples)
# This will be useful to testing convergence
old_tau = np.inf

# ---- Let's Roll! ----

# Now we'll sample for up to n_samples steps
for sample in sampler.sample(Start_Location, iterations=n_samples, progress=True):
    # Only check convergence every 100 steps
    if sampler.iteration % Check_Interv:
        continue

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1

    h5f=h5py.File(ConvergeFile, 'w')
    h5f.create_dataset('autocorr', data=autocorr)
    h5f.create_dataset('index', data=index)
    h5f.close()

    # Check convergence
    converged = np.all(tau * Converge_Thresh < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 1/Converge_Thresh)
    if converged:
        break
    old_tau = tau
