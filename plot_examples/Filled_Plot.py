# Get 1D plot

LineWidth = 2
FontSize = 15
Level = 1

import matplotlib.pyplot as plt
import numpy as np
T_arcade = lambda TR, Beta, v: TR*v**Beta # define model
# Data
B = -2.62
TR = 1.19
SigmaB = 0.04
SigmaT = 0.14
B1 = B - Level*SigmaB
B2 = B + Level*SigmaB
TR1 = TR - Level*SigmaT
TR2 = TR + Level*SigmaT

z = np.arange(13,35,0.1)
v = v = 1.42/(1+z)
T1 = T_arcade(TR1, B2, v)
T2 = T_arcade(TR2, B1, v)
T = T_arcade(TR, B, v)

plt.rcParams.update({'font.family':'Times'})
plt.rcParams['text.usetex'] = True

plt.fill_between(z, T1, T2,color = 'b',alpha=0.3,label='Confidence Region')
plt.plot(z,T,color = 'k',linestyle='-',linewidth=LineWidth,label = 'Best Fit')
plt.xlabel('$z$',fontsize=FontSize,fontname='Times')
plt.ylabel('$T_{\mathrm{excess}}$',fontsize=FontSize,fontname='Times')
plt.legend(fontsize=FontSize,loc = 'upper left')
plt.xticks(size=FontSize)
plt.yticks(np.arange(0,8000,1000),size=FontSize)
plt.title('Arcade Radio Excess',fontsize=FontSize)
plt.tight_layout()
plt.show()
