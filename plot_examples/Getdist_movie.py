reload = 1
reload_status = 0
n_frames = 220
dpi = 500
head_frame = 1000
LineWidth = 2
FontSize = 18
gif_file = '/Users/cangtao/Desktop/Pop_III_Posterior.gif'
Root = '/Users/cangtao/cloud/Library/PyLab/plot_examples/data/29_MCG_HII_50/29_MCG_HII_50_'
Use_dynamic_Range = 1

# ---- Initialise ----
import getdist, os, shutil, copy
from getdist import plots
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from PIL import Image
from PyLab import *

t1 = TimeNow()

Root_2 = Root + '_movie_tmp'
def Get_New_Chains(idx = 0, n_fram = 100, head = 1000):
    if idx > n_fram:
        raise Exception('Frame index exceeds n_fram')
    CF1 = Root + '.txt'
    CF2 = Root_2 + '.txt'
    RF1 = Root + '.ranges'
    RF2 = Root_2 + '.ranges'
    NF1 = Root + '.paramnames'
    NF2 = Root_2 + '.paramnames'
    if not os.path.exists(NF2):
        shutil.copy(NF1, NF2)
    if not os.path.exists(RF2):
        shutil.copy(RF1, RF2)
    C1 = np.loadtxt(CF1)
    ChainShape = np.shape(C1)
    CL1 = ChainShape[0]
    
    dN = CL1 - head
    dn = int(np.round(dN/n_fram))
    CL2 = head + idx * dn
    CL2 = min(CL2, CL1)
    if idx == n_frames-1:
        CL2 = CL1
    C2 = np.zeros((CL2, ChainShape[1]))
    for idx1 in np.arange(0, CL2):
        for idx2 in np.arange(0, ChainShape[1]):
            C2[idx1, idx2] = C1[idx1, idx2]
    np.savetxt(CF2, C2, fmt='%.8E', delimiter='  ')
    PlotFile = Root + '_' + str(idx) + '.png'
    return PlotFile

if Use_dynamic_Range:
    RF = Root + '.ranges_nameless'
    r0 = Getdist_Marg_Stat(Root = Root, Param_Name = 'fR_mini')
    r1 = Getdist_Marg_Stat(Root = Root, Param_Name = 'L_X_MINI')
    r2 = Getdist_Marg_Stat(Root = Root, Param_Name = 'F_STAR7_MINI')
    r3 = Getdist_Marg_Stat(Root = Root, Param_Name = 'F_ESC7_MINI')
    r4 = Getdist_Marg_Stat(Root = Root, Param_Name = 'A_LW')
    Low = 'low_3'
    Top = 'upper_3'
    Prior = np.loadtxt(RF)
    
    def Get_Plot_Range(idx, side, param_idx):
        if param_idx == 0:
            R2 = r0
        elif param_idx == 1:
            R2 = r1
        elif param_idx == 2:
            R2 = r2
        elif param_idx == 3:
            R2 = r3
        elif param_idx == 4:
            R2 = r4
        R1 = Prior[param_idx,:]
        if side == -1:
            y1 = R1[0]
            y2 = max(R2[Low], y1)
        else:
            y1 = R1[1]
            y2 = min(R2[Top], y1)
        x1 = 0
        x2 = n_frames
        y = (y2 - y1)*(idx - x1)/(x2 - x1) + y1

        return y
    '''
    param = 4
    idx1 = Get_Plot_Range(200, -1, param)
    idx2 = Get_Plot_Range(200, 1, param)

    print(idx1, idx2)
    '''
    
def Compute_IMG(idx):
    show_status_info(idx, n_frames, 1000, 'Compute_IMG', 1)
    
    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.family':'Times'})
    g = plots.getSubplotPlotter(subplot_size = 3)
    g.settings.axes_fontsize=14
    g.settings.title_limit_fontsize = 14
    g.settings.lab_fontsize =14
    PlotFile = Get_New_Chains(idx = idx, n_fram = n_frames, head = head_frame)
    samples = getdist.mcsamples.loadMCSamples(Root_2)
    p = samples.getParams()
    LGD = 'Status = ' + "{:.1f}".format(100*idx/n_frames) + '\%'
    g.triangle_plot(
        [samples],
        width_inch=12,
        contour_colors=['blue'],
        legend_labels=['EDGES + Arcade + $\\tau_{\mathrm{rei}}$'],
        filled = True,
        line_args=[{'lw':1.5,'ls':'-', 'color':'k'}],
        param_limits = {'fR_mini': [Get_Plot_Range(idx, -1, 0), Get_Plot_Range(idx, 1, 0)], 
                        'L_X_MINI' : [Get_Plot_Range(idx, -1, 1), Get_Plot_Range(idx, 1, 1)],
                        'F_STAR7_MINI' : [Get_Plot_Range(idx, -1, 2), Get_Plot_Range(idx, 1, 2)], 
                        'F_ESC7_MINI' : [Get_Plot_Range(idx, -1, 3), Get_Plot_Range(idx, 1, 3)], 
                        'A_LW' : [Get_Plot_Range(idx, -1, 4), Get_Plot_Range(idx, 1, 4)]}, # set axis limits
        title_limit=2)
    plt.savefig(PlotFile, dpi = dpi)

'''
plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots()
plt.rcParams.update({'font.family':'Times'})
g = plots.getSubplotPlotter(subplot_size = 3)
g.settings.axes_fontsize=14
g.settings.title_limit_fontsize = 14
g.settings.lab_fontsize =14

if reload:
    for idx in np.arange(0, n_frames):
        print('----', idx/n_frames)
        PlotFile = Get_New_Chains(idx = idx, n_fram = n_frames, head = head_frame)
        samples = getdist.mcsamples.loadMCSamples(Root_2)
        p = samples.getParams()
        LGD = 'Status = ' + "{:.1f}".format(100*idx/n_frames) + '\%'
        g.triangle_plot(
            [samples],
            width_inch=12,
            contour_colors=['blue'],
            legend_labels=['EDGES + Arcade + $\\tau_{\mathrm{rei}}$'],
            filled = True,
            line_args=[{'lw':1.5,'ls':'-', 'color':'k'}],
            param_limits = {'fR_mini': [Get_Plot_Range(idx, -1, 0), Get_Plot_Range(idx, 1, 0)], 
                            'L_X_MINI' : [Get_Plot_Range(idx, -1, 1), Get_Plot_Range(idx, 1, 1)],
                            'F_STAR7_MINI' : [Get_Plot_Range(idx, -1, 2), Get_Plot_Range(idx, 1, 2)], 
                            'F_ESC7_MINI' : [Get_Plot_Range(idx, -1, 3), Get_Plot_Range(idx, 1, 3)], 
                            'A_LW' : [Get_Plot_Range(idx, -1, 4), Get_Plot_Range(idx, 1, 4)]}, # set axis limits
            title_limit=2)
        plt.savefig(PlotFile, dpi = dpi)
'''
if reload:
    for idx in np.arange(0, n_frames):
        Compute_IMG(idx)

# Status Bar
def Write_Status(idx, get_name = 0):
    show_status_info(idx, n_frames, 100, name = 'Write_Status', show_status=1)
    NewPlotFile = Root + '_status_' + str(idx) + '.png'
    plt.rcParams.update({'font.family':'Times'})
    plt.rcParams['text.usetex'] = True
    if get_name:
        return NewPlotFile
    fig, ax = plt.subplots()
    PlotFile = Get_New_Chains(idx = idx, n_fram = n_frames, head = head_frame)
    img = Image.open(PlotFile)
    ax.imshow(img)
    IMG_Shape = np.shape(img)
    nx = IMG_Shape[0]
    ny = IMG_Shape[1]
    xmin = nx/10
    xmax = nx * 0.9
    x2 = xmax
    dx = xmax - xmin
    
    CF = Root + '.txt'
    Chain = np.loadtxt(CF)
    ChainShape = np.shape(Chain)
    ChainLen = ChainShape[0]
    status = (head_frame + idx * (ChainLen - head_frame)/n_frames)/ChainLen
    status_str = "{:.1f}".format(100*status) + '\%'
    if idx == n_frames-1:
        status_str = '100\%'
    x1 = xmin + status * dx
    y1 = ny*0.985
    y2 = ny*0.965
    
    if idx != n_frames-1:
        ax.fill_between([x1, x2], y1, y2, color = 'grey')
    ax.plot([xmin, xmax], [y1, y1], 'grey', linewidth = 0.5)
    ax.plot([xmin, xmax], [y2, y2], 'grey', linewidth = 0.5)
    ax.plot([xmin, xmin], [y1, y2], 'grey', linewidth = 0.5)
    ax.plot([xmax, xmax], [y1, y2], 'grey', linewidth = 0.5)
    
    plt.text(xmax*1.01, y1, status_str, size=8, rotation=0, color='k')
    plt.xlim([0, nx])
    plt.ylim([ny, 0])
    
    plt.tight_layout()
    plt.axis('off')  # Remove axis ticks and labels
    plt.savefig(NewPlotFile, bbox_inches = 'tight', dpi = dpi, pad_inches = 0)
    return NewPlotFile

images = []
for idx in np.arange(0, n_frames):
    if reload_status:
        PlotFile = Write_Status(idx, 0)
    else:
        PlotFile = Write_Status(idx, 1)
    # PlotFile = Get_New_Chains(idx = idx, n_fram = n_frames, head = head_frame)
    img = Image.open(PlotFile)
    images.append(img)
    #img = copy.deepcopy(Image.open(PlotFile))
    #images.append(img)
images[0].save(gif_file, save_all=True, append_images=images[1:], duration=5, loop=1)
Timer(t1)

'''
IMG = []
for idx in np.arange(0, n_frames):
    if reload_status:
        PlotFile = Write_Status(idx, 0)
    else:
        PlotFile = Write_Status(idx, 1)
    IMG.append(imageio.imread(PlotFile))

# Save the images as a GIF
output_gif_path = "output.gif"
imageio.mimsave(output_gif_path, IMG, duration=0.5)
'''
