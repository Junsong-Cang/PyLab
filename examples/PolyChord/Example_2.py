from numpy import pi, log, sqrt
import pypolychord
import numpy as np
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior

nDims = 3
nDerived = 0
p1_range = {'min':-10,'max':10, 'start':1}
p2_range = {'min':-10,'max':10, 'start':1}
p3_range = {'min':-10,'max':10, 'start':1}

TxtChainFile = "data/Example_polychord_JSC.txt"
Getdist_Range_File = "data/Example_polychord.range"

# ---- Get Fake Data ----
p1=2
p2=-3
p3=1.2

def Model(p1,p2,p3,x):
	return p1+p2*x+p3*x**2

x_data=np.arange(0,3,0.1)
y_data=Model(p1,p2,p3,x_data)
sigma_data=0.2 + 0.4*np.random.rand(len(x_data))

def Squeeze(Range,HyperParam):
  '''Return physical param value for HyperParam in [-1,1]'''
  y1 = Range['min']
  y2 = Range['max']
  return (y2-y1)*(HyperParam+1)/2+y1

# ---- Define Likelihood and Prior ----
def log_likelihood(theta):
  'This is lnL'
  global x_data, y_data, sigma_data, p1_range, p2_range, p3_range
  hp1, hp2, hp3 = theta
  p1 = Squeeze(p1_range,hp1)
  p2 = Squeeze(p2_range,hp2)
  p3 = Squeeze(p3_range,hp3)
  y=Model(p1,p2,p3,x_data)
  LnL = -0.5 * np.sum(((y-y_data)**2)/(sigma_data**2))
  # Add prior here
  if np.abs(hp1) < 1.0 and np.abs(hp2) < 1.0 and np.abs(hp3) < 1.0:
    return LnL,[0.0]
  else:
    return -np.inf,[0.0]

'''
def likelihood(theta):
    """ Simple Gaussian Likelihood"""

    nDims = len(theta)
    r2 = sum(theta**2)
    logL = -log(2*pi*sigma*sigma)*nDims/2.0
    logL += -r2/2/sigma/sigma

    return logL, [r2]

'''

def prior(hypercube):
    """ Uniform prior from [-1,1]^D. """
    return UniformPrior(-1, 1)(hypercube)


#| Optional dumper function giving run-time read access to
#| the live points, dead points, weights and evidences

def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])

#| Initialise the settings

settings = PolyChordSettings(nDims, nDerived)
settings.file_root = 'Example_polychord'
settings.nlive = 200
settings.do_clustering = True
settings.read_resume = False

#| Run PolyChord

output = pypolychord.run_polychord(
  loglikelihood = log_likelihood, 
  nDims = nDims, 
  nDerived = nDerived, 
  settings = settings, 
  prior = prior,
  dumper = dumper)

'''
#| Create a paramnames file

paramnames = ['p1  p_1', 'p2  p_2', 'p3  p_3']
output.make_paramnames_files(paramnames)

#| Make an anesthetic plot (could also use getdist)
try:
    from anesthetic import NestedSamples
    samples = NestedSamples(root= settings.base_dir + '/' + settings.file_root)
    fig, axes = samples.plot_2d(['p1','p2','p3'])
    fig.savefig('posterior.pdf')

except ImportError:
    try:
        import getdist.plots
        posterior = output.posterior
        g = getdist.plots.getSubplotPlotter()
        g.triangle_plot(posterior, filled=True)
        g.export('posterior.pdf')
    except ImportError:
        print("Install matplotlib and getdist for plotting examples")

    print("Install anesthetic or getdist  for for plotting examples")
'''