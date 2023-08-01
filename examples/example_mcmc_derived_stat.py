
LineWidth = 2
FontSize = 15
Level = 2

reload = 0
nv = 100
Root = '/Users/cangtao/cloud/GitHub/Radio_Excess_EDGES/Example_pymultinest/chains/arcade_'

from PyLab import *
import matplotlib.pyplot as plt

v_array = np.logspace(-1.7, 1.51, nv)
def model(theta):
  # Excess Temp
  TR = theta[1]
  Beta = theta[2]
  T = TR*v_array**Beta
  return T
if reload:
  r = mcmc_derived_stat(
    model_function = model,
    FileRoot = Root
  )
  np.savez('tmp.npz', data = r)
else:
  r = np.load('tmp.npz')['data']

Mean = r[0][:]
L1 = r[1][:]
U1 = r[2][:]
L2 = r[3][:]
U2 = r[4][:]

plt.rcParams.update({'font.family':'Times'})
plt.rcParams['text.usetex'] = True

# ----Fitting Results----
if Level == 1:
  plt.fill_between(v_array, L1, U1,color = 'b',alpha=0.3,label='Confidence Region')
else:
  plt.fill_between(v_array, L2, U2,color = 'b',alpha=0.3,label='Confidence Region')
plt.loglog(v_array, Mean,color = 'k',linestyle='-',linewidth=LineWidth,label = 'Best Fit')

# ----data points----
v = np.array([0.022, 0.045, 0.408, 1.42, 3.2, 3.41, 7.97, 8.33, 9.72, 10.49, 29.5, 31, 90])
t = np.array([21200, 4355, 16.24, 3.213, 2.792, 2.771, 2.765, 2.741, 2.732, 2.732, 2.529, 2.573, 2.706])
SigmaT = np.array([5125, 520, 3.4, 0.53, 0.01, 0.009, 0.014, 0.016, 0.006, 0.006, 0.155, 0.076, 0.019])

plt.errorbar(v,t-2.728,SigmaT,color = 'k',linewidth=LineWidth,label = 'Data Points',fmt='+')

plt.xlabel('$\\nu$ [GHz]',fontsize=FontSize,fontname='Times')
plt.ylabel('$T_{\mathrm{excess}}$',fontsize=FontSize,fontname='Times')
plt.legend(fontsize=FontSize,loc = 'lower left')
plt.xticks(size=FontSize)
plt.yticks(size=FontSize)
plt.title('Arcade Radio Excess',fontsize=FontSize)
plt.tight_layout()

plt.show()
