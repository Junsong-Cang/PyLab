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
ConvergeFile='data/Status.h5'

# ---- Get Fake Data ----
p1=2
p2=-3
p3=0.5

def Model(p1,p2,p3,x):
	return p1+p2*x+p3*x**2

x_data=np.arange(0,3,0.1)
y_data=Model(p1,p2,p3,x_data)
sigma_data=0.2 + 0.4*np.random.rand(len(x_data))

# ---- Define Likelihood and Prior ----
def log_likelihood(theta):
	'This is lnL'
	global x_data, y_data, sigma_data
	p1, p2, p3 = theta
	y=Model(p1,p2,p3,x_data)
	return -0.5 * np.sum(((y-y_data)**2)/(sigma_data**2))

def log_prior(theta):
	p1, p2, p3 = theta
	global p1_range, p2_range, p3_range
	if p1_range['min'] < p1 < p1_range['max'] and p2_range['min'] < p2 < p2_range['max'] and p3_range['min'] < p3 < p3_range['max']:
		return 0.0
	else:
		return -np.inf

def log_probability(theta):
    global TxtChainFile, PrintGetdist
    LogP = log_prior(theta) + log_likelihood(theta)
    F=open(TxtChainFile,'a')
    Wight = 0.9999
    p1, p2, p3 = theta
    Chi2 = -2 * LogP
    print("{0:.5E}".format(Wight), "    {0:.5E}".format(Chi2), "    {0:.5E}".format(p1), "    {0:.5E}".format(p2), "    {0:.5E}".format(p3), file=F)
    F.close()
    return LogP, [0.0]

# ----
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
    # print(UniformPrior(-1, 1)(hypercube))
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
  loglikelihood = log_probability, 
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