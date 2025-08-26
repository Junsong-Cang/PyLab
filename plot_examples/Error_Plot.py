LineWidth = 2
FontSize = 18

import matplotlib.pyplot as plt
import numpy as np
v = np.array([0.022, 0.045, 0.408, 1.42, 3.2, 3.41, 7.97, 8.33, 9.72, 10.49, 29.5, 31, 90])
t = np.array([21200, 4355, 16.24, 3.213, 2.792, 2.771, 2.765, 2.741, 2.732, 2.732, 2.529, 2.573, 2.706])
SigmaT = np.array([5125, 520, 3.4, 0.53, 0.01, 0.009, 0.014, 0.016, 0.006, 0.006, 0.155, 0.076, 0.019])
Fit = 2.729 + 1.19 *v**-2.62

plt.rcParams.update({
    'text.usetex': True,
    'font.family':'Times',
    'text.latex.preamble': r'\usepackage{newtxtext,newtxmath}'})
fig, ax = plt.subplots()
ax.grid(True, which='both', linewidth = 0.3)  # `which='both'` enables major and minor grids

# use default fmt if want to connect datapoints
plt.errorbar(v,t,SigmaT,color = 'k',linewidth=LineWidth,label = 'Data Points',fmt='+')
# For asymetric error bar use this:
# plt.errorbar(v,t,[Sigma_Low, Sigma_Top],color = 'k',linewidth=LineWidth,label = 'Data Points',fmt='+')

plt.plot(v,Fit,color = 'k',linestyle='-',linewidth=LineWidth,label = 'Best Fit')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\\nu$ [GHz]',fontsize=FontSize,fontname='Times New Roman')
plt.ylabel('$T$ [K]',fontsize=FontSize,fontname='Times New Roman')
plt.xticks(size=FontSize)
plt.yticks(size=FontSize)
plt.legend(fontsize=FontSize,loc = 'upper right')
plt.tight_layout()
plt.savefig('/Users/cangtao/Desktop/tmp.pdf')
