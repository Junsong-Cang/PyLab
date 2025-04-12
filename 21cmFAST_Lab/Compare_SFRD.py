
import numpy as np
File = 'data/SFRD_box_EOS.txt'

def Read_SFRD(File = 'data/SFRD_box_EOS.txt', f10 = 0.1, f7 = 0.1, aX = 1, model = 1):
    d = np.loadtxt(File, skiprows=1)
    print(d)
    pass

Read_SFRD()
