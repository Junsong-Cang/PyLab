'''
Some useful functions
- Read_Curve
- print_mcmc_info
- Getdist_Marg_Stat
- mcmc_derived_stat
- Find_Index
- Interp_3D
- LCDM_HyRec
- Is_Scalar
- Function_Array
- Integrate
- Hubble
- Find_Negative_Element
- Signal_HyperCube
- Map
- Solve
- MCR
- TimeNow
- Timer
- Interp_2D
- Within_Range
- Get_dydx
- show_status_info
- SaySomething
- HyRec
- Trim_Axis : get same x or y axis
- Read_Curve_GUI : read curve now with GUI
- derived_param_chains : 
- Resize_chains : 
- Find_dif : find difference between two arrays
- Set_Uniform_Multinest_Prior
- Get_Numerical_Passwd
- Get_Chain_stat
- Read_MultiNest_BestFit
- Read_MultiNest_Stats
- h5disp
'''

import numpy as np
import shutil, os, time, warnings, h5py
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from joblib import Parallel, delayed
from PIL import Image
from scipy import interpolate
import matplotlib.pyplot as plt
try:
    import tqdm
except:
    pass
try:
    import getdist
    from getdist import plots
except:
    pass
from matplotlib.backend_bases import MouseEvent

# First let's define some exceptions
class PyLab_Solve_Exception_Small(Exception):
    # Too small
    def __init__(self, f1, f2, x1, x2, count):
        print('Exception in PyLab.Solve: Cannot find solution for x in ', [x1, x2], ', [f1, f2] =', [f1, f2],', iteration =', count)
class PyLab_Solve_Exception_Large(Exception):
    # Too large
    def __init__(self, f1, f2, x1, x2, count):
        print('Exception in PyLab.Solve: Cannot find solution for x in ', [x1, x2], ', [f1, f2] =', [f1, f2],', iteration =', count)
class PyLab_Range_Error(Exception):
    def __init__(self, xmin, xmax, x, name):
        print('Range error for', name, ', axis range:', [xmin, xmax], ', input:', x)

def Read_Curve(
    File = '/Users/cangtao/cloud/Library/PyLab/Curve_Data/Jaehong_Park_18_SFRD.txt',
    nx = 100,
    model = 1,
    Convert_x = False,
    Convert_y = False):
    '''
    Extract curve data
    ---- inputs ----
    File : File Name
    nx : number of x
    model : What to use as output
            1 - spline interpolation
            2 - linear interpolation
            3 - no interpolation
            4 - raw pixel data
    Convert_x : convert extracted x to 10^x
    Convert_y : convert extracted y to 10^y
    '''
    warnings.warn('Read_Curve will be deappreciated soon, please use Read_Curve_GUI or MCR in the future')

    data = np.loadtxt(File)
    # Get anchor, r - real, p - pixel
    xr1 = data[0]
    xp1 = data[2]
    xr2 = data[4]
    xp2 = data[6]
    yr1 = data[9]
    yp1 = data[11]
    yr2 = data[13]
    yp2 = data[15]
    
    # Retrieving data
    Len = np.rint((len(data)-16)/2).astype(int)
    xp = np.linspace(0,1,Len)
    yp = np.linspace(0,1,Len)
    xr = np.linspace(0,1,Len)
    yr = np.linspace(0,1,Len)
    for id in np.arange(0, Len):
        xid = 16 + 2*id
        xp[id] = data[xid]
        yp[id] = data[xid+1]
        xr[id] = (xr2 - xr1)*(xp[id] - xp1)/(xp2 - xp1) + xr1
        yr[id] = (yr2 - yr1)*(yp[id] - yp1)/(yp2 - yp1) + yr1
    # Adjust x order: small to large
    if xr1 > xr2:
        xr = xr[-1:0:-1]
        yr = yr[-1:0:-1]
    # Set your outputs
    if model == 4:
        # Raw pixel info
        x, y = xp, yp
    elif model == 3:
        # Physical info with no interpolation
        x, y = xr, yr
    elif model == 2:
        x = np.linspace(xr[0], xr[-1], nx)
        y = np.interp(x, xr, yr)
    elif model == 1:
        x = np.linspace(xr[0], xr[-1], nx)
        y = spline(xr, yr)(x)
    # Convert from log10
    if model != 4:
        if Convert_x:
            x = pow(10,x)
        if Convert_y:
            y = pow(10,y)
    return x,y

def Read_Curve_Pro(
    File = '/Users/cangtao/cloud/Library/PyLab/Curve_Data/2306_17136_fig1a_ipta.txt',
    nx = 100,
    model = 3,
    Convert_x = False,
    Convert_y = False):
    
    swap_file = '/tmp/tmp_curve_reader.txt'
    
    data = np.loadtxt(File)
    head = data[0:16]
    # print(head)
    nan_idx = np.where(np.isnan(data))[0]
    r = []

    for idx in np.arange(0, len(nan_idx)):
        id1 = nan_idx[idx] + 1
        if idx == len(nan_idx) - 1:
            id2 = len(data)
        else:
            id2 = nan_idx[idx+1]
        
        curve = np.zeros((16 + id2 - id1, 1))
        curve[0:16,0] = head[:]
        curve[16:len(curve),0] = data[id1:id2]
        np.savetxt(fname = swap_file, X = curve, fmt = '%.8E', delimiter = '')
        x, y = Read_Curve(
            File = swap_file,
            nx = nx,
            model = model,
            Convert_x = Convert_x,
            Convert_y = Convert_y)
        r_here = {'x' : x, 'y' : y}
        r.append(r_here)

    return r
    
def print_mcmc_info(FileRoot, info):
    '''
    Print parameter names and ranges for getdist, useful for pymultinest post-processing
    ----inputs----
    FileRoot: File root
    info: mcmc run info, a list of form [p1, p2, ...]
          p1 and p2 etc are dictionaries with these keys: 'name', 'min', 'max', 'latex'
    '''
    NameFile = FileRoot + '.paramnames'
    RangeFile = FileRoot + '.ranges'
    n = len(info)
    
    nf=open(NameFile,'w')
    rf=open(RangeFile,'w')
    for id in np.arange(0,n):
        param = info[id]
        print(param['name'],'	',param['latex'], file = nf)
        print(param['name'],'	',param['min'], '   ', param['max'],file = rf)
    nf.close()
    rf.close()

def Getdist_Marg_Stat(Root = '/Users/cangtao/cloud/GitHub/Radio_Excess_EDGES/data/0_EDGES/0_EDGES_',
                      Param_Name = 'fR',
                      n_sigma = 4):
  '''
  Get the marginalised stat of a param
  '''
  s = getdist.mcsamples.loadMCSamples(Root)
  stats = s.getMargeStats()
  lim = stats.parWithName(Param_Name).limits
  Low_1=lim[0].lower
  Low_2=lim[1].lower
  Top_1=lim[0].upper
  Top_2=lim[1].upper
  mean = s.mean(Param_Name)

  # Ok also get estimate for 3 and 4 Sigmas
  
  Sigma = (Top_2 - Low_2)/4
  Center = (Top_2 + Low_2)/2
  Low_3 = Center - 3*Sigma
  Top_3 = Center + 3*Sigma
  Low_n = Center - n_sigma*Sigma
  Top_n = Center + n_sigma*Sigma

  r = {'mean':mean, 'low_68':Low_1, 'low_95':Low_2, 'upper_68':Top_1, 'upper_95':Top_2,
       'low_3' : Low_3, 'low_n' : Low_n, 'upper_3' : Top_3, 'upper_n' : Top_n}

  return r

def mcmc_derived_stat(
    model_function = lambda x: np.sum(x),
    FileRoot = '/Users/cangtao/cloud/GitHub/Radio_Excess_EDGES/data/5_UVLF/5_UVLF_',
    NewRoot = '_derived_params',
    cache_loc='/tmp/',
    cleanup = True,
    print_status = False,
    reload_sample = 1,
    sample_fraction = 2,
    ncpu = 1,
    prior_min = np.array([np.nan, np.nan]),
    prior_max = np.array([np.nan, np.nan]),
    remove_NaN = 0):
    '''
    Get statistics for derived params, currently not compatible with mpi CosmoMC
    example can be found in examples/example_mcmc_derived_stat.py
    ----inputs----
    model_function:
        a function which uses original params to get derived param
    FileRoot:
        Original mcmc chain file root
    cache_loc:
        location to cache new chains
    cleanup:
        whether to clean up files produced during the call
    Use_PreComputed_Samples:
        Use pre-computed samples stored in PreComputed_Samples
    ----outputs----
    an array of form:
    Result[0][:] = mean
    Result[1][:] = low_68
    Result[2][:] = upper_68
    Result[3][:] = low_95
    Result[4][:] = upper_95
    Result[5][:] = MAP
    '''
    t1 = TimeNow()
    if NewRoot[0] == '/':
        # raise Exception('NewRoot cannot begin with /')
        New_Root = NewRoot
    else:
        New_Root = FileRoot + NewRoot

    if reload_sample:
        derived_param_chains(
            model = model_function,
            old_root = FileRoot,
            new_root = New_Root,
            get_plot = 0,
            plot_file = '/Users/cangtao/Desktop/tmp.png', 
            derived_names = None,
            clean_up = False,
            prior_min = None,
            prior_max = None,
            ncpu = ncpu,
            show_status = print_status)
    RawChain = np.loadtxt(FileRoot+'.txt')
    RawSize = np.shape(RawChain)
    DerivedChain = np.loadtxt(New_Root + '.txt')
    Cache_Root = cache_loc+'tmp_mcmc_post_processing' if cache_loc[-1]=='/' else cache_loc+'/tmp_mcmc_post_processing'

    # copy and add name
    NF0 = New_Root + '.paramnames'
    Cache_Name_File = Cache_Root + '.paramnames'
    shutil.copy(NF0, Cache_Name_File)
    nf=open(Cache_Name_File, 'a')
    Name = 'derived_param*     p_{derived}'
    print(Name, file = nf)
    nf.close()
    
    n_deriv = np.shape(DerivedChain)[1] - RawSize[1]
    def get_new_samples(param_idx):
        # Ok let's parallise this thing
        if sample_fraction < 0.999:
            Chain_Len = RawSize[0]
            New_Len = int(np.round(sample_fraction * Chain_Len))
        else:
            New_Len = RawSize[0]
        #NewChain = np.zeros((New_Len, RawSize[1] + 1))
        NewChain = np.zeros([New_Len, RawSize[1] + 1])
        
        # First get chains
        for idx1 in np.arange(0, New_Len):
            for idx2 in np.arange(0, RawSize[1] + 1):
                if idx2 < RawSize[1]:
                    NewChain[idx1, idx2] = DerivedChain[idx1, idx2]
                else:
                    NewChain[idx1, idx2] = DerivedChain[idx1, RawSize[1] + param_idx]
        
        # Now save to files
        PreFix = Cache_Root + '_swap_' + str(param_idx)
        ChainFile = PreFix + '.txt'
        np.savetxt(ChainFile, NewChain, fmt='%.8E', delimiter='  ')
        Range_File = New_Root + '.ranges'
        Range_File_Here = PreFix + '.ranges'
        shutil.copy(Range_File, Range_File_Here)
        Name_File_Here = PreFix + '.paramnames'
        shutil.copy(Cache_Name_File, Name_File_Here)

        # Files ready, now get stats
        # stat = Getdist_Marg_Stat(Root = PreFix, Param_Name = 'derived_param')
        try:
            stat = Getdist_Marg_Stat(Root = PreFix, Param_Name = 'derived_param')
        except:
            # This can happen if observable is tightly confined, e.g for xH when xH=0 (neutral) or xH=1 (ionised)
            warnings.warn('Getdist failed to compute statistics for this parameter, returning NaN or nearest element, status now:')
            print(param_idx/n_deriv)
            stat = {'mean': np.nan, 'low_68': np.nan, 'upper_68': np.nan, 'low_95': np.nan, 'upper_95': np.nan}
        
        r = np.zeros(5)
        r[0] = stat['mean']
        r[1] = stat['low_68']
        r[2] = stat['upper_68']
        r[3] = stat['low_95']
        r[4] = stat['upper_95']
        
        # Clean up
        os.remove(ChainFile)
        os.remove(Range_File_Here)
        os.remove(Name_File_Here)

        return r
    
    #r = np.zeros((5, n_deriv))
    r = np.zeros([6, n_deriv]) # Added feature: MAP
    MAP = model_function(Read_MultiNest_Stats(Root=FileRoot)['MAP'])
    r[5,:] = MAP[:]
    
    if ncpu == 1:
        for pid in np.arange(0, n_deriv):
            show_status_info(idx = pid, n = n_deriv, nx = 50, name = 'get_derived_stat_vec', show_status = print_status)
            swap = get_new_samples(pid)
            r[0:5, pid] = swap[:]
    else:
        swap = Parallel(n_jobs = ncpu)(delayed(get_new_samples)(pid) for pid in np.arange(0, n_deriv))
        swap = np.array(swap)
        for pid in np.arange(0, n_deriv):
            for idx_2 in np.arange(0, 5):
                r[idx_2, pid] = swap[pid, idx_2]
    
    # Determine whether to refine for prior and NaN
    Refine = 0
    if remove_NaN:
        Refine = 1
    
    if Is_Scalar(prior_min):
        prior_min_vec = prior_min * np.ones(n_deriv)
    else:
        prior_min_vec = prior_min
    if Is_Scalar(prior_max):
        prior_max_vec = prior_max * np.ones(n_deriv)
    else:
        prior_max_vec = prior_max

    if False in np.isnan(prior_min_vec):
        Refine = 1
        refine_prior_min = 1
    else:
        refine_prior_min = 0
    
    if False in np.isnan(prior_max_vec):
        Refine = 1
        refine_prior_max = 1
    else:
        refine_prior_max = 0
    
    def Remove_NaN_Elements(x):
        # replace all NaN element with nearest non-nan
        NaN_Finder = np.isnan(x)
        NaN_idx = []
        OK_idx = []
        nx = len(x)
        for idx in np.arange(0, nx):
            if NaN_Finder[idx]:
                NaN_idx.append(idx)
            else:
                OK_idx.append(idx)
        if len(NaN_idx) != 0:
            new_x = np.zeros(nx)
            NaN_idx = np.array(NaN_idx)
            OK_idx = np.array(OK_idx)
            
            for idx in np.arange(0, nx):
                if not np.isnan(x[idx]):
                    new_x[idx] = x[idx]
                else:
                    dist_vec = (idx - OK_idx)**2
                    dist_min_idx = np.argmin(dist_vec)
                    new_x_idx = int(np.round(OK_idx[int(np.round(dist_min_idx))]))
                    new_x[idx] = x[new_x_idx]
        else:
            new_x = x
        return new_x

    if Refine:
        if remove_NaN:
            for idx in np.arange(0, 5):
                x0 = r[idx,:]
                x1 = Remove_NaN_Elements(x0)
                for pid in np.arange(0, n_deriv):
                    r[idx, pid] = x1[pid]
        
        for idx in np.arange(0, 5):
            for pid in np.arange(0, n_deriv):
                if not np.isnan(r[idx,pid]):

                    if refine_prior_min:
                        if r[idx,pid] < prior_min_vec[pid]:
                            r[idx,pid] = prior_min_vec[pid]

                    if refine_prior_max:
                        if r[idx,pid] > prior_max_vec[pid]:
                            r[idx,pid] = prior_max_vec[pid]
    
    if cleanup:
        os.remove(New_Root + '.txt')
        os.remove(New_Root + '.paramnames')
        os.remove(New_Root + '.ranges')
    
    if print_status:
        print('Time used for mcmc_derived_stat:', TimeNow() - t1)
    return r

def Find_Index(
        x = 0.5,
        x_axis = np.linspace(0, 10, 11)):
    '''
    Find left neibouring index, useful for interpolation
    '''
    id1 = 0
    id3 = len(x_axis) - 1
    Go = 1
    
    x_is_reversed = 0
    x_vec = x_axis

    if x_vec[0] > x_vec[-1]:
        x_is_reversed = 1
        x_vec = x_axis[::-1]
        # raise Exception('x_vec must be increasing')
    if x < x_vec[0] or x > x_vec[-1]:
        print('--------crash imminent--------')
        print('x not in range, debug info:')
        print('x_min = ', x_vec[0], ', x = ', x, ', x_max = ', x_vec[-1])
        raise Exception('Range error in Finx_Index')
    
    count = 0
    while Go:
        count = count + 1
        id2_ = id1 + (id3 - id1)/2
        id2 = np.round(id2_).astype(int)
        x2 = x_vec[id2]
        if x <= x2:
            id3 = id2
        else:
            id1 = id2
        if id3 - id1 < 2:
            Go = 0
        if count > 10000:
            print('--------crash imminent--------')
            print('I donno why this is happening, count is too large')
            raise Exception
    
    # print('x1 = ', x_vec[id1], ', x = ', x, ', x2 = ', x_vec[id2], ', x3 = ', x_vec[id3]) # debug info
        
    if x_is_reversed:
        id1 = len(x_axis) - 1 - id3

    # print('x1 = ', x_axis[id1], ', x = ', x, ', x2 = ', x_axis[id1 + 1], ', x3 = ', x_axis[id1 + 2]) # debug info
    
    return id1

"""
def Interp_3D(Tab, x_axis, y_axis, z_axis, x, y, z, Use_Extrap = False):
    '''
    Interpolate a 3D array
    Tab: [xid, yid, zid]
    '''
    
    x_ = x
    y_ = y
    z_ = z

    if Use_Extrap:
        if (x_axis[0] > x_axis[-1]) or (y_axis[0] > y_axis[-1]) or (z_axis[0] > z_axis[-1]):
            raise Exception('axis must be accending when Use_Extrap')
        
        if x_ < x_axis[0]:
            x_ = x_axis[0]
        elif x_ > x_axis[-1]:
            x_ = x_axis[-1]
        
        if y_ < y_axis[0]:
            y_ = y_axis[0]
        elif y_ > y_axis[-1]:
            y_ = y_axis[-1]
        
        if z_ < z_axis[0]:
            z_ = z_axis[0]
        elif z_ > z_axis[-1]:
            z_ = z_axis[-1]
    
    # do x-axis first
    id1 = Find_Index(x_, x_axis)
    id2 = id1 + 1
    x1 = x_axis[id1]
    x2 = x_axis[id2]
    f1 = Tab[id1,:,:]
    f2 = Tab[id2,:,:]
    f = (f2 - f1)*(x_ - x1)/(x2 - x1) + f1
    
    # next the y-axis
    id1 = Find_Index(y_, y_axis)
    id2 = id1 + 1
    y1 = y_axis[id1]
    y2 = y_axis[id2]
    g1 = f[id1,:]
    g2 = f[id2,:]
    g = (g2 - g1)*(y_ - y1)/(y2 - y1) + g1

    # finally the z axis
    id1 = Find_Index(z_, z_axis)
    id2 = id1 + 1
    z1 = z_axis[id1]
    z2 = z_axis[id2]
    r1 = g[id1]
    r2 = g[id2]
    r = (r2 - r1)*(z_ - z1)/(z2 - z1) + r1

    return r

"""

def Interp_3D(Tab, x_axis, y_axis, z_axis, x, y, z, Use_Extrap = False):
    '''
    Interpolate a 3D array
    Tab: [xid, yid, zid]
    '''
    
    x_ = x
    y_ = y
    z_ = z

    if Use_Extrap:
        if (x_axis[0] > x_axis[-1]) or (y_axis[0] > y_axis[-1]) or (z_axis[0] > z_axis[-1]):
            raise Exception('axis must be accending when Use_Extrap')
        
        if x_ < x_axis[0]:
            x_ = x_axis[0]
        elif x_ > x_axis[-1]:
            x_ = x_axis[-1]
        
        if y_ < y_axis[0]:
            y_ = y_axis[0]
        elif y_ > y_axis[-1]:
            y_ = y_axis[-1]
        
        if z_ < z_axis[0]:
            z_ = z_axis[0]
        elif z_ > z_axis[-1]:
            z_ = z_axis[-1]
    
    # do x-axis first
    id1 = Find_Index(x_, x_axis)
    id2 = id1 + 1
    x1 = x_axis[id1]
    x2 = x_axis[id2]
    f1 = Tab[id1]
    f2 = Tab[id2]
    f = (f2 - f1)*(x_ - x1)/(x2 - x1) + f1
    
    # next the y-axis
    id1 = Find_Index(y_, y_axis)
    id2 = id1 + 1
    y1 = y_axis[id1]
    y2 = y_axis[id2]
    g1 = f[id1]
    g2 = f[id2]
    g = (g2 - g1)*(y_ - y1)/(y2 - y1) + g1

    # finally the z axis
    id1 = Find_Index(z_, z_axis)
    id2 = id1 + 1
    z1 = z_axis[id1]
    z2 = z_axis[id2]
    r1 = g[id1]
    r2 = g[id2]
    r = (r2 - r1)*(z_ - z1)/(z2 - z1) + r1

    return r

def LCDM_HyRec(z = np.logspace(0,2,10), Use_EoR = False):
    # Get xe and Tk for LCDM using template generated by HyRec
    from Useful_Numbers import LCDM_Recombination_Template, EOS21_EoR_Template
    if Use_EoR:
        Template = EOS21_EoR_Template
    else:
        Template = LCDM_Recombination_Template
    lgzp_axis = Template['log10_zp']
    lt_axis = Template['log10_Tk']
    lx_axis = Template['log10_xe']

    Size = np.shape(z)
    if Size == ():
        # z is a scalar
        lgzp_ = np.log10(1+z)
        if lgzp_ < lgzp_axis[0]:
            xe = lx_axis[0]
            Tk = lt_axis[0]
        elif lgzp_ > lgzp_axis[-1]:
            xe = lx_axis[-1]
            Tk = np.log10(2.728 * (1+z))
        else:
            xe = spline(lgzp_axis, lx_axis)(lgzp_)
            Tk = spline(lgzp_axis, lt_axis)(lgzp_)
        xe, Tk = 10**xe, 10**Tk
        return xe, Tk
    else:
        nz = Size[0]
    # Work on array
    xe = np.linspace(0,1,nz)
    Tk = np.linspace(0,1,nz)
    lgzp = np.log10(1+z)
    for idx in np.arange(0,nz):
        lgzp_ = lgzp[idx]
        if lgzp_ < lgzp_axis[0]:
            # if z too low: use minimum
            xe[idx] = lx_axis[0]
            Tk[idx] = lt_axis[0]
        elif lgzp_ > lgzp_axis[-1]:
            xe[idx] = lx_axis[-1]
            Tk[idx] = np.log10(2.728*(1+z[idx]))
        else:
            xe[idx] = spline(lgzp_axis, lx_axis)(lgzp_)
            Tk[idx] = spline(lgzp_axis, lt_axis)(lgzp_)
    xe, Tk = 10**xe, 10**Tk
    return xe, Tk

def Is_Scalar(x, ErrorMethod = 0):
    '''
    ----inputs----
    ErrorMethod : what to do if detected array or scalar
                0 - nothing
                1 - Raise error if found scalar
                2 - Raise error if found array
    '''
    Size = np.shape(x)
    if Size == ():
        r = True
        if ErrorMethod == 1:
            raise Exception('Scalar detected')
    else:
        r = False
        if ErrorMethod == 2:
            raise Exception('Array detected')
    return r

def Function_Array(f, x):
    '''
    Get y array for a scalar function
    ----inputs----
    f : function which takes scalar as input
    x : array
    '''
    Is_Scalar(x=x, ErrorMethod=1)
    nx = len(x)
    r = np.linspace(0,1,nx)
    for idx in np.arange(0,nx):
        r[idx] = f(x[idx])
    return r

def Integrate(f = lambda x,y: np.sin(x), 
              y1 = 0, 
              x1 = 0, 
              x2 = np.pi,
              nx = 100,
              show_status = False):
    '''
    Integrate dy/dx from x1 to x2 with nx inetrvals
    ----inputs----
    f    : a function, f(x,y) returns dy/dx, both x and y are scalar
    y1   : initial condition, y@x1
    x1   : starting point
    x2   : end point
    nx   : number of x interval
    '''
    x = np.linspace(x1, x2, nx)
    dx = (x2-x1)/(nx-1)
    y = np.linspace(0,1,nx)
    y[0] = y1
    for idx in np.arange(1,nx):
        x_ = x[idx]
        dydx = f(x_, y[idx-1])
        dy = dydx * dx
        y[idx] = y[idx-1] + dy
        if show_status:
            print('Status = ', idx/nx, ', x = ', x_, ', y = ', y[idx])
    return x,y

def Hubble(z=0, OmM = 0.30964168161, h = 0.6766, OmR = 9.1E-5):
    '''
    Hubble rate for LCDM model
    '''
    OmL = 1 - OmM - OmR
    H0 = h*3.240755744239557E-18
    zp3 = pow(1+z, 3)
    zp4 = pow(1+z, 4)
    r = H0 * np.sqrt(OmL + OmM * zp3 + OmR * zp4)
    return r

def Find_Negative_Element(x = np.array([1,2,-2,3]), Small = 1e-200, model = 1):
    r = 0
    Xmin = np.min(x)
    if Xmin < -Small:
        r = 1
        if model == 1:
            raise Exception('Found negative element, Xmin = ', Xmin)
    return r


def Signal_HyperCube(model = lambda x: np.sum(x) * np.linspace(0,1,10),
                     ParamCube = np.random.rand(3,6),
                     Print_Status = True,
                     ncpu = 1):
    '''
    ParamCube : [mean, error_low, error_up]
    '''
    dim = np.shape(ParamCube)[1]
    if (dim != 6) and (dim != 3):
        raise Exception('Current version only support dim = 3 and dim = 6')
    cube = np.empty(np.shape(ParamCube))
    '''
    for idx in np.arange(0,dim):
        cube[0,idx] = ParamCube[0,idx]
        cube[1,idx] = cube[0,idx] - ParamCube[1,idx]
        cube[2,idx] = cube[0,idx] + ParamCube[2,idx]
    '''
    cube[0] = ParamCube[0]
    cube[1] = ParamCube[0] - ParamCube[1]
    cube[2] = ParamCube[0] + ParamCube[2]
    '''
    print(ParamCube)
    print(cube)
    '''

    medium = model(cube[0,:])
    Signal_Size=len(medium)
    r = np.empty((3,Signal_Size))
    r[0,:] = medium[:]
    params = []
    if dim == 6:
        for id1 in [1,2]:
            for id2 in [1,2]:
                for id3 in [1,2]:
                    for id4 in [1,2]:
                        for id5 in [1,2]:
                            for id6 in [1,2]:
                                params.append([cube[id1,0], cube[id2,1], cube[id3,2], cube[id4,3], cube[id5,4], cube[id6,5]])
    elif dim == 3:
        for id1 in [1,2]:
            for id2 in [1,2]:
                for id3 in [1,2]:
                    params.append([cube[id1,0], cube[id2,1], cube[id3,2]])

    if ncpu == 1:
        samples = []
        for idx in np.arange(0,2**dim):
            param = params[idx]
            new_sample = model(param)
            samples.append(new_sample)
            if Print_Status:
                print('Status for Signal_HyperCube: ', idx/2**dim)
    else:
        samples = Parallel(n_jobs = ncpu)(delayed(model)(x) for x in params)
    
    y = np.linspace(0,1,2**dim)

    for idx in np.arange(0,Signal_Size):
        for id in np.arange(0,2**dim):
            y[id] = samples[id][idx]
        r[1][idx] = np.min(y)
        r[2][idx] = np.max(y)
    
    return r

def Map(
        F = lambda x: np.exp(-(x-10)**2), 
        Start = 1,
        Width = 1,
        MinX = -3,
        MaxX = 4,
        nx = 100,
        Precision = 1e-2,
        Max_Iteration = 50,
        Use_log_x = 1,
        flat_count = 10,
        Print_debug_MSG = False):
    '''
    Find converging profile of function F
    ----inputs----
    F : function of form F(x), x MUST be allowed to be a vector
    Start : starting location, preferably peak, in log10
    Width : Width in log10
    MinX : Minimum search region, in log10
    MaxX : Maximum search region, in log10
    nx : timesteps per Width
    Precision : When to stop
    Max_Iteration : maximum iteration
    Use_log_x : whether to check convergence using {\int dlnx f} or {\int dx f}, preferably true because sampling is done in log
    '''

    convergence_fraction = 0.2

    Small = 1e-200
    if MinX >= MaxX:
        raise Exception('Setting error, MinX is larger than MaxX, MinX = ', MinX, ', MaxX = ', MaxX)
    
    # Step 1: Find lx_left
    Found_Left = 0
    count = 0
    count_tot = 0
    while not Found_Left:
        '''
        When to stop:
        1. found lx_left
        2. count overflow
        '''
        if count == 0:
            lx1 = Start - Width
            lx2 = Start
        else:
            lx2 = lx1
            lx1 = lx1 - Width
        if lx1 < MinX:
            Found_Left = 1
            lx1 = MinX
        x = np.logspace(lx1, lx2, nx)
        x = np.delete(x, [0, -1]) # avoid repeated elements
        f = F(x)
        Find_Negative_Element(x = f)
        if count == 0:
            x_vec = x
            f_vec = f
        else:
            x_vec = np.concatenate((x, x_vec))
            f_vec = np.concatenate((f, f_vec))
            # Checking convergence using the left 20% (by default) samples
            nx_now = len(x_vec)
            idx_2 = np.round(nx_now * convergence_fraction).astype(int)
            if idx_2 < 3:
                raise Exception('Not enough sample to check convergence, you need to increase nx')
            x_cvg = x_vec[0:idx_2] # cvg is for convergence
            f_cvg = f_vec[0:idx_2]
            if Use_log_x:
                Int_tot = np.trapz(x = np.log(x_vec), y = f_vec)
                Int_left = np.trapz(x = np.log(x_cvg), y = f_cvg)
            else:
                Int_tot = np.trapz(x = x_vec, y = f_vec)
                Int_left = np.trapz(x = x_cvg, y = f_cvg)
            
            if Int_tot < Small or Int_left < Small:
                # could be that f_vec is flat on the left, I am gonna allow that for flat_count iterations
                if count > flat_count:
                    Found_Left = 1
            else:
                dif = Int_left/Int_tot
                if dif < Precision:
                    Found_Left = 1
                    count_tot = count
                    if Print_debug_MSG:
                        print('-------- lx_left found --------')
                        print('Int_tot = ', Int_tot)
                        print('Int_left = ', Int_left)
                        print('dif = ', dif)
                        print('count_tot = ', count_tot)
                        print('x_left_now = ', np.min(x_vec))

            if count > Max_Iteration:
                Found_Left = 1
                print('MSG from Map.Find_Left: no solution found after allowed iterations, debug info:')
                print('lx1 = ', lx1)
                print('lx2 = ', lx2)
                print('count = ', count)
                raise Exception('Max_Iteration exceeded in Find_Left, debug info: lx1 = ', lx1, ', lx2 = ', lx2, ', count = ', count)
            
        count = count + 1
        
    # Step 2: Find lx_right
    Found_Right = 0
    count = 0
    while not Found_Right:
        '''
        When to stop:
        1. found lx_right
        2. count overflow
        '''
        if count == 0:
            lx1 = Start
            lx2 = Start + Width
        else:
            lx1 = lx2
            lx2 = lx1 + Width
        if lx2 > MaxX:
            Found_Right = 1
            lx2 = MaxX
        x = np.logspace(lx1, lx2, nx)
        x = np.delete(x, [0, -1]) # avoid repeated elements
        f = F(x)
        Find_Negative_Element(x = f)
        
        x_vec = np.concatenate((x_vec, x))
        f_vec = np.concatenate((f_vec, f))
        
        nx_now = len(x_vec)
        idx_1 = nx_now - np.round(nx_now * convergence_fraction).astype(int)

        if nx_now - idx_1 < 5:
            raise Exception('Not enough sample to check convergence, you need to increase nx')
        x_cvg = x_vec[idx_1:nx_now - 1]
        f_cvg = f_vec[idx_1:nx_now - 1]

        if Use_log_x:
            Int_right = np.trapz(x = np.log(x_cvg), y = f_cvg)
            Int_tot = np.trapz(x = np.log(x_vec), y = f_vec)
        else:
            Int_right = np.trapz(x = x_cvg, y = f_cvg)
            Int_tot = np.trapz(x = x_vec, y = f_vec)
        
        if Int_right < Small or Int_tot < Small:
            # could be that f_vec is flat on the right, I am gonna allow that for first 10 iterations
            if count > 10:
                Found_Right = 1
        else:
            dif = Int_right/Int_tot
            if dif < Precision:
                Found_Right = 1
                count_tot = count_tot + count
                if Print_debug_MSG:
                        print('-------- lx_right found --------')
                        print('Int_tot = ', Int_tot)
                        print('Int_right = ', Int_right)
                        print('dif = ', dif)
                        print('count_tot = ', count_tot)
                        print('x_right_now = ', np.max(x_vec))
                
        if count > Max_Iteration:
            print('MSG from Map.Find_Right: no solution found after allowed iterations, debug info:')
            print('lx1 = ', lx1)
            print('lx2 = ', lx2)
            print('count = ', count)
            raise Exception('Max_Iteration exceeded in Find_Right, debug info: lx1 = ', lx1, ', lx2 = ', lx2, ', count = ', count)
            
        count = count + 1
        
    # You can do a booster run now if you like, but that's too much work for now

    return x_vec, f_vec

def Solve(
        F = lambda x:x**3 - 3,
        Xmin = 0,
        Xmax = 10,
        Precision = 1e-3,
        show_status = 0):
    '''
    Find solution for F(x) = 0
    '''
    Small = 1e-200
    CountMax = 10000
    x1 = Xmin
    x2 = Xmax
    x_ = x1 + (x2 - x1)/2
    
    Proceed = 1
    count = 0
    
    f1 = F(x1)
    f2 = F(x2)
        
    while Proceed:
        count = count + 1
        f_ = F(x_)
        if f1*f2 > 0:
            Proceed = 0
            if f1 > 0:
                raise PyLab_Solve_Exception_Large(f1,f2,x1,x2,count)
            else:
                raise PyLab_Solve_Exception_Small(f1,f2,x1,x2,count)
        
        # Eliminate rare event
        if np.abs(f1) < Small:
            r = x1
            Proceed = 0
        elif np.abs(f2) < Small:
            r = x2
            Proceed = 0
        elif np.abs(f_) < Small:
            r = x_
            Proceed = 0
        
        # update x1, x2 and f1, f2
        if f1*f_ < 0:
            # solution is in [x1, x_]
            x1 = x1
            x2 = x_
            f1 = f1
            f2 = f_
        else:
            # solution is in [x_, x2]
            x1 = x_
            x2 = x2
            f1 = f_
            f2 = f2

        x_ = x1 + (x2 - x1)/2
        dif = np.abs(x2-x1)
        
        # Check convergence
        if x1 > Small:
            dif = dif/x1
            if dif < Precision:
                Proceed = 0
                r = x_
        else:
            # This is to avoid NaN or Inf error from dif/0
            if dif < Precision:
                Proceed = 0
                r = x_
        
        # The rest is debug section

        if count > CountMax:
            Proceed = 0
            r = 1e-200/1e-200
            print('Crash iminent, debug info:')
            print('x1 = ', x1, 'x_ = ', x_, 'x2 = ', x2)
            print('f1 = ', f1, 'f_ = ', f_, 'f2 = ', f2)
            print('count = ', count)
            
            raise Exception('Solution not found within permitted counts')
        
        if show_status:
            print('count = ', count, ', dif = ', "{0:.4E}".format(dif), ', [x1, x2] = [',  "{0:.4E}".format(x1), "{0:.4E}".format(x2),']')
            
    return r

def MCR(
        Filename = '/Users/cangtao/Desktop/chains/chains_11/11_Ann_yy_PLK_mcr.png',
        xmin = -5,
        xmax = 3.7,
        ymin = 0,
        ymax = 20,
        nx = 200,
        mode = 1,
        Smooth_nx = 100,
        Convert_x = 0,
        Convert_y = 0,
        y_unit = 1):
    '''
    Extract x and y data from a getdist plot image
    ---- inputs ----
    Filename : name of input image file
    xmin : left boundary of input image
    xmax : right boundary of input image
    ymin : lower boundary of input image
    ymax : upper boundary of input image
    nx   : number of x in extracted data
    mode : what you want as outputs
           1 : extracted data, spline interpoaltion
           2 : extracted data, linear interpoaltion
           3 : extracted data, no interpolation
           4 : raw pixel data, useful for debugging
    Smooth_nx : number of x in smoothening downsample, no smoothening if Smooth_nx < 2
    Convert_x : whether or not convert x to 10**x
    Convert_y : whether or not convert y to 10**y
    '''

    if mode not in [1, 2, 3, 4]:
        raise Exception("Wrong choice of mode")

    im = Image.open(Filename)
    width = im.size[0]
    height = im.size[1]
    im = im.convert('RGB')

    # Step 1 : Get pixel locations (x1, y1)
    x1 = np.arange(0, width)
    y1 = np.empty(width)
    for xid in range(0, width):
        R_top, G_top, B_top = im.getpixel((xid, 0)) # RGB info for top region
        yid = 1
        PROCEED = 1
        while PROCEED == 1:
            R, G, B = im.getpixel((xid, yid))
            Same_Color = ((R == R_top) & (G == G_top) & (B == B_top))
            if Same_Color:
                yid = yid + 1
            else:
                PROCEED = 0
        y1[xid] = yid
    
    # No need to proceed if we have what we want already
    if mode == 4:
        return x1, y1
    
    # Step 2 : Get physical locations (x2, y2)
    x2 = np.linspace(xmin, xmax, width)
    y2 = np.empty(width)
    for i in range(0, width):
        y2[i] = (height - y1[i]) * (ymax - ymin) / height + ymin

    if mode == 3:
        return x2, y2

    # Step 3 : Get low-res samples for smoothened results
    if Smooth_nx < 2:
        x3, y3 = x2, y2
    else:
        x3 = np.linspace(xmin, xmax, Smooth_nx)
        y3 = interpolate.interp1d(x2, y2)(x3)

    # Step 4 : interpolate
    x4 = np.linspace(xmin, xmax, nx)
    if mode == 1:
        y4 = interpolate.interp1d(x3, y3, kind='cubic')(x4)
    elif mode == 2:
        y4 = interpolate.interp1d(x3, y3)(x4)
    
    # Post-processing
    if Convert_x:
        x4 = 10**x4
    if Convert_y:
        y4 = 10**y4
    y4 = y4 * y_unit
    
    return x4, y4

def TimeNow():
    return time.time()

def Timer(t1):
    dt = TimeNow() - t1
    print('Time used :', dt)
    return dt

def Interp_2D(Tab, x_axis, y_axis, x_target, y_target, Use_Log_X = False, Use_Log_Y = False, Use_Log_Z = False):
    '''
    Interpolate a 2D array
    Tab: [xid, yid]
    '''
    
    # Current version only support scalar targets
    Is_Scalar(x_target, 2)
    Is_Scalar(y_target, 2)

    # print('Tab = ', np.shape(Tab), ', x = ', np.shape(x_axis), ', y = ', np.shape(y_axis))
    
    # Choose whether to do interpolation in log space, abort logspace if found 0
    Small = 1e-200

    if Use_Log_X and np.min(x_axis) > Small:
        x_vec = np.log10(x_axis)
        x = np.log10(x_target)
    else:
        x_vec = x_axis
        x = x_target
    if Use_Log_Y and np.min(y_axis) > Small:
        y_vec = np.log10(y_axis)
        y = np.log10(y_target)
    else:
        y_vec = y_axis
        y = y_target
    if Use_Log_Z:
        Fxy = np.log10(Tab)
        Fxy[Fxy<-150]=-150
        Fxy_is_log = 1
    else:
        Fxy = Tab
        Fxy_is_log = 0

    # do x-axis first
    id1 = Find_Index(x, x_vec)
    id2 = id1 + 1
    tmp_1, tmp_2 = id1, id2
    x1 = x_vec[id1]
    x2 = x_vec[id2]
    f1 = Fxy[id1, :]
    f2 = Fxy[id2, :]
    Fy = (f2 - f1)*(x - x1)/(x2 - x1) + f1
    
    # now do y-axis
    id1 = Find_Index(y, y_vec)
    id2 = id1 + 1
    
    '''
    # debug section
    f11 = Tab[tmp_1, id1]
    f12 = Tab[tmp_1, id2]
    f21 = Tab[tmp_2, id1]
    f22 = Tab[tmp_2, id2]
    
    print('zid1 = ', tmp_1, ', zid2 = ', tmp_2)
    print('mid1 = ', id1, ', mid2 = ', id2)
    print('f11 = ', "{0:.6E}".format(f11), ', f12 = ', "{0:.6E}".format(f12), ', f21 = ', "{0:.6E}".format(f21), ', f22 = ', "{0:.6E}".format(f22))
    
    x1 = np.log(x_axis[tmp_1])
    x2 = np.log(x_axis[tmp_2])
    x_ = np.log(x_target)
    y1 = np.log(y_axis[id1])
    y2 = np.log(y_axis[id2])
    y_ = np.log(y_target)
    f11 = np.log(f11)
    f12 = np.log(f12)
    f21 = np.log(f21)
    f22 = np.log(f22)
    
    f1 = (f12 - f11)*(y_ - y1)/(y2 - y1) + f11
    f2 = (f22 - f21)*(y_ - y1)/(y2 - y1) + f21
    f = (f2 - f1)*(x_ - x1)/(x2 - x1) + f1
    print('recovered result python: ', np.exp(f))
    '''
    
    y1 = y_vec[id1]
    y2 = y_vec[id2]
    F1 = Fy[id1]
    F2 = Fy[id2]
    r = (F2 - F1)*(y - y1)/(y2 - y1) + F1

    if Fxy_is_log:
        r = 10**r
    return r

def Within_Range(x, x_array):
    '''
    Check whether x is within the range of x_array,
    might be useful for interpolation
    return True if in range
    '''
    
    xmin = np.min(x_array)
    xmax = np.max(x_array)
    if x < xmin or x > xmax:
        r = False
    else:
        r = True
    return r

def Get_dydx(
        x = np.linspace(0,1, 100),
        y = np.sin(np.linspace(0,1, 100)),
        Use_log = False,
        method = 0
    ):
    '''
    Get dy/dx from array
    ------inputs----
    x : x array
    y : y array
    Use_log : do derivative in log space, might give better precision if data is linear-log
    method : 
        0 - dydx[i] = (y[i+1]-y[i])/(x[i+1]-x[i])
        1 - dydx[i] = (y[i+1]-y[i-1])/(x[i+1]-x[i-1])
        2 - use spline extended samples and get dydx from method 1
    '''

    def get_dydx_kernel(v , f, model):
        nv = len(v)
        r = np.zeros(nv)
        if model == 0:
            for idx in np.arange(0, nv-1):
                dv = v[idx+1] - v[idx]
                df = f[idx+1] - f[idx]
                r[idx] = df/dv
            r[nv-1] = r[nv-2]
        elif model == 1:
            for idx in np.arange(1, nv-1):
                dv = v[idx+1] - v[idx-1]
                df = f[idx+1] - f[idx-1]
                r[idx] = df/dv
            r[0] = r[1]
            r[nv-1] = r[nv-2]
        return r
    
    if Use_log:
        v = np.log(x)
        f = np.log(y)
    else:
        v = x
        f = y
    
    if method == 0 or method == 1:
        r = get_dydx_kernel(v = v, f = f, model = method)
    elif method == 2:
        v2 = np.linspace(v[0], v[-1], 10000)
        f2 = spline(v, f)(v2)
        '''
        plt.plot(v,f, 'k')
        plt.plot(v2,f2, '--r')
        plt.show()
        '''
        r2 = get_dydx_kernel(v = v2, f = f2, model = 1)
        r = np.interp(x = v, xp = v2, fp = r2)

    if Use_log:
        # r is now dlny/dlnx, convert now
        r = r*x/y
    return r

def SaySomething(
        File = '/Users/cangtao/Desktop/tmp.txt',
        MSG = '----'):
    '''
    Say something to the File, useful for MPI status monitor
    '''
    cmd = 'echo ' + MSG + ' >> ' + File
    os.system(cmd)

def HyRec(
    Pann = 1e-28,
    Use_SSCK = 0,
    DM_Channel = 0,
    mdm = 100.0):
    if os.getenv('COSMOMC_PATH') is not None:
        HyRec_Path = os.getenv('COSMOMC_PATH')+'HyRec/'
    else:
        raise Exception('Canot find HyRec executable, please specify COSMOMC_PATH env variable, or you can use LCDM result with LCDM_HyRec')
    inputs = [
        '----DM_Parameters----',
        'Use_SSCK',
        str(int(Use_SSCK)),
        'DM_Channel',
        str(int(DM_Channel+1)),
        'Mdm',
        "{0:.8E}".format(mdm),
        'Pann',
        "{0:.8E}".format(Pann),
        'Gamma',
        '0',
        '----PBH_Parameters----',
        'PBH_Model',
        '1',
        'PBH_Distribution',
        '1',
        'Mbh',
        '1E3',
        'fbh',
        '0',
        'PBH_Lognormal_Sigma',
        '1',
        'PBH_PWL_Mmax',
        '1.0E16',
        'PBH_PWL_Gamma',
        '0.5',
        'PBH_Spin',
        '0',
        '----Cosmological_Parameters----',
        'Tcmb',
        '2.728',
        'obh2',
        '.02242',
        'omh2',
        '.1424',
        'okh2',
        '0',
        'odeh2',
        '0.31538756',
        'w0,wa',
        '-1 0',
        'YHe',
        '.245',
        'Neff',
        '3.046']
    
    IF = 'tmp_in.txt'
    OF = 'tmp_out.txt'
    IF_Full = HyRec_Path + IF
    F=open(IF_Full,'w')
    for s in inputs:
        print(s, file=F)
    F.close()

    cmd = 'cd ' + HyRec_Path + ';./hyrec<' + IF + '>'+OF
    os.system(cmd)
    OF_Full = HyRec_Path + OF
    d = np.loadtxt(OF_Full)
    z = d[:,0]
    xe = d[:,1]
    Tk = d[:,2]
    os.remove(OF_Full)
    os.remove(IF_Full)

    r = {'z' : z, 'xe' : xe, 'Tk' : Tk}

    return r

def derived_param_chains(
        model = lambda x: np.sum(x), 
        old_root = '/Users/cangtao/cloud/GitHub/Radio_Excess_EDGES/data/28_EDGES_tau/28_EDGES_tau_', 
        new_root = '/Users/cangtao/cloud/GitHub/Radio_Excess_EDGES/data/28_EDGES_tau/28_EDGES_tau_derived', 
        get_plot = 0,
        plot_file = '/Users/cangtao/Desktop/tmp.pdf', 
        derived_names = {'name': 'tau', 'latex': '\\tau_{\mathrm{rei}}'},
        clean_up = False,
        prior_min = None,
        prior_max = None,
        ncpu = 1,
        write_names = True,
        show_status = 0):
    '''
    Create a new set of chains for derived quantities
    ---inputs----
    model : get derived params from active params, should be of the same format as model_function in mcmc_derived_stat,
        it goes as something like derived_params = model(x), here x is the vector of active params
    old_root : old getdist fileroot
    new_root : file root of new samples, if begins with '/' then it's used, otherwise joined with old root
    get_plot : get a triangular plot for new samples? applicable to only one derived param
    plot_file : if get_plot, where to store plot
    derived_names : a list of dictionaries containing 'name' and 'latex' keys
    clean_up : delete derived chains after producing the getdist plot
    prior_min : minimum of prior, can be array or scalar
    prior_max : maximum of prior, can be array or scalar
    ncpu : number of cpus for computing the new samples
    show_status : whether or not to show status
    '''
    
    t1 = TimeNow()
    CF0 = old_root + '.txt'
    chain_0 = np.loadtxt(CF0)
    chain_shape= np.shape(chain_0)
    chain_len = chain_shape[0]
    chain_width = chain_shape[1]

    # determine derived dimension using first sample
    p0 = chain_0[0, 2:]
    sample_p0 = model(p0)
    if Is_Scalar(sample_p0, 0):
        n_deriv = 1
    else:
        n_deriv = len(sample_p0)
    
    ActiveParams = chain_0[:, 2:]
    
    def Get_New_Samples(idx):
        # Get derived params for a given index in active param
        # show_status_info(idx, chain_len, 1000, name = 'derived_param_chains', show_status = show_status)
        param = ActiveParams[idx]
        r = model(param)
        return r

    if ncpu == 1:
        # DerivedSamples = np.zeros((chain_len, n_deriv))
        if n_deriv == 1:
            DerivedSamples = np.zeros(chain_len)
            #for idx in np.arange(0, chain_len):
            for idx in tqdm.tqdm(range(chain_len), desc = 'Computing sampels', disable = not show_status):
                DerivedSamples[idx] = Get_New_Samples(idx)
        else:
            DerivedSamples = np.zeros((chain_len, n_deriv))
            # for idx in np.arange(0, chain_len):
            for idx in tqdm.tqdm(range(chain_len), desc = 'Computing sampels', disable = not show_status):
                NewSample = Get_New_Samples(idx)
                for pid in np.arange(0, n_deriv):
                    DerivedSamples[idx, pid] = NewSample[pid]
    else:
        DerivedSamples = Parallel(n_jobs=ncpu)(delayed(Get_New_Samples)(idx) for idx in tqdm.tqdm(range(chain_len), desc = 'Computing sampels', disable = not show_status))
        DerivedSamples = np.array(DerivedSamples)
    NewChain = np.zeros((chain_len, chain_width + n_deriv))
    for idx in np.arange(0, chain_len):
        for pid in np.arange(0, chain_width + n_deriv):
            if pid < chain_width:
                NewChain[idx, pid] = chain_0[idx, pid]
            else:
                if n_deriv == 1:
                    NewChain[idx, pid] = DerivedSamples[idx]
                else:
                    NewChain[idx, pid] = DerivedSamples[idx, pid - chain_width]
    # Saving New Chain
    if new_root[0] == '/':
        NewRoot = new_root
    else:
        NewRoot = old_root + new_root
    CF1 = NewRoot + '.txt'
    np.savetxt(CF1, NewChain, fmt='%.8E', delimiter='  ')

    # Get Param Names and Ranges
    NF0 = old_root + '.paramnames'
    NF1 = NewRoot + '.paramnames'
    shutil.copy(NF0, NF1)
    
    RF0 = old_root + '.ranges'
    RF1 = NewRoot + '.ranges'
    shutil.copy(RF0, RF1)
    
    '''
    if derived_names != None:
        if len(derived_names) != n_deriv:
            warnings.warn('length of derived_names does not match length of derived parameters, using default settings for param names')
    '''
    
    if n_deriv == 1:
        if write_names:
            nf=open(NF1,'a')
            Name = derived_names['name'] + '*     ' + derived_names['latex']
            print(Name, file = nf)
            nf.close()

        if prior_min != None or prior_max != None:
            # Some times we only know one side of the prior
            if prior_min != None:
                p_min = "{0:.5E}".format(prior_min)
            else:
                p_min = '-Inf'
            if prior_max != None:
                p_max = "{0:.5E}".format(prior_max)
            else:
                p_max = 'Inf'
            RF=open(RF1,'a')
            RangeInfo = derived_names['name'] + '*     ' + p_min + '    ' + p_max
            print(RangeInfo, file = RF)
            RF.close()
    else:
        if write_names:
            if n_deriv < 10:
                nf=open(NF1,'a')
                for param_id in np.arange(0, n_deriv):
                    Name = derived_names['name'][param_id] + '*     ' + derived_names['latex'][param_id]
                    print(Name, file = nf)
                nf.close()
            else:
                warnings.warn('cannot write names for n_deriv > 10')

    if get_plot:
        if n_deriv < 12:
            # Get some very basic plots
            plt.rcParams['text.usetex'] = True
            plt.rcParams.update({'font.family':'Times'})
            samples = getdist.mcsamples.loadMCSamples(NewRoot)
            p = samples.getParams()
            g = plots.getSubplotPlotter(subplot_size = 3)
            g.settings.axes_fontsize=14
            g.settings.title_limit_fontsize = 14
            g.triangle_plot(
                samples,
                width_inch=12,
                contour_colors=['blue'],
                filled = True,
                line_args=[{'lw':1.5,'ls':'-', 'color':'k'}],
                title_limit=2)
            plt.tight_layout()
            plt.savefig(plot_file, dpi=500)
            print('Getdist plot saved to :')
            print(plot_file)
        else:
            warnings.warn('Too many derived params for a triangular plot')

    if clean_up:
        os.remove(CF1)
        os.remove(NF1)
        os.remove(RF1)
    if show_status:
        print('Time used for derived_param_chains: ', time.time() - t1)

def Trim_Axis(x1, y1, x2, y2, nx):
    '''
    For a pair of x, y array, convert to the same axis
    '''
    
    x_min = max(x1.min(), x2.min())
    x_max = min(x1.max(), x2.max())
    x_new = np.linspace(x_min, x_max, nx)
    if x1[-1] > x1[0]:
        y1_new = np.interp(x = x_new, xp = x1, fp = y1)
    else:
        y1_new = np.interp(x = x_new, xp = x1[::-1], fp = y1[::-1])
    if x2[-1] > x2[0]:
        y2_new = np.interp(x = x_new, xp = x2, fp = y2)
    else:
        y2_new = np.interp(x = x_new, xp = x2[::-1], fp = y2[::-1])
    return x_new, y1_new, y2_new

def Read_Curve_GUI(
        x1 = 1,
        x2 = 100,
        y1 = 1e-3,
        y2 = 1e2,
        IMG_File = '/Users/cangtao/cloud/Library/PyLab/data/Read_Curve_GUI_Example.png',
        OutputFile = '/tmp/tmp_Read_Curve_GUI_data.npz',
        LogX = False,
        LogY = False,
        show_plot = False,
        x_pix_min = 100,
        show_cursor = 1
    ):
    '''
    A GUI interface for extracting plot data
    ---- inputs ----
        x1 : left anchor
        x2 : right anchor
        y1 : bottom anchor
        y2 : top anchor
        IMG_File : image file
        OutputFile : npz file to store data
        LogX : is x axis in log scale
        LogY : is y axis in log scale
        show_plot : show comparison plot between input IMG and extracted data
        x_pix_min : minimum x pixel, below which a click ends the function
        show_cursor : show cursor with a cross
    '''
    warnings.warn('Read_Curve_GUI: argument model is deappreciated')

    # Define a cross hair class which shows mouse cursor as cross, we are loading a 2D image so it's gonna be slow
    class Cross_hair_Cursor:
        def __init__(self, ax):
            self.ax = ax
            self.horizontal_line = ax.axhline(color='k', lw=0.3, ls='-')
            self.vertical_line = ax.axvline(color='k', lw=0.3, ls='-')
            # text location in axes coordinates
            self.text = ax.text(0.9, 0.1, '', transform=ax.transAxes)

        def set_cross_hair_visible(self, visible):
            need_redraw = self.horizontal_line.get_visible() != visible
            self.horizontal_line.set_visible(visible)
            self.vertical_line.set_visible(visible)
            self.text.set_visible(visible)
            return need_redraw

        def on_mouse_move(self, event):
            if not event.inaxes:
                need_redraw = self.set_cross_hair_visible(False)
                if need_redraw:
                    self.ax.figure.canvas.draw()
            else:
                self.set_cross_hair_visible(True)
                x, y = event.xdata, event.ydata
                # update the line positions
                self.horizontal_line.set_ydata([y])
                self.vertical_line.set_xdata([x])
                self.text.set_text(f'x={x:1.2f}, y={y:1.2f}')
                self.ax.figure.canvas.draw()
    
    # display clicked points
    def onclick(event):
        if event.inaxes == ax:  # Check if the mouse click is within the plot axes
            ax.plot(event.xdata, event.ydata, 'r+', linewidth = 0.2)  # Plot the clicked points as red crosses
            plt.draw()

    fig, ax = plt.subplots()
    # Enable interactive mode
    plt.ion()

    # Load the PNG image
    image = Image.open(IMG_File)
    ax.imshow(image)

    # indicate stop regions
    IMG_Size = np.shape(image)
    Y_Size = IMG_Size[0]
    x_boundary = x_pix_min*np.ones(Y_Size)
    y_boundary = np.linspace(0, Y_Size, Y_Size)
    ax.plot(x_boundary, y_boundary, '--r')

    plt.axis('off')  # Remove axis ticks and labels
    if show_cursor:
        cursor = Cross_hair_Cursor(ax)
        fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
    plt.xlim([0, IMG_Size[1]])
    plt.ylim([IMG_Size[0], 0])
    
    if np.abs(x1) < 1000:
        print('click on x = ', "{:.2f}".format(x1), ':')
    else:
        print('click on x = ', "{0:.3E}".format(x1), ':')

    plt.show()

    #Track mouse click locations
    x_pixel = []
    y_pixel = []
    
    proceed = 1
    count = 0
    while proceed:
        # Wait for a mouse click event
        clicks = plt.ginput(n=1, timeout=0)
        # Check if a mouse click occurred
        if count == 0:
            if np.abs(x2) < 1000:
                print('click on x = ', "{:.2f}".format(x2), ':')
            else:
                print('click on x = ', "{0:.3E}".format(x2), ':')
        elif count == 1:
            if np.abs(y1) < 1000:
                print('click on y = ', "{:.2f}".format(y1), ':')
            else:
                print('click on y = ', "{0:.3E}".format(y1), ':')
        elif count == 2:
            if np.abs(y2) < 1000:
                print('click on y = ', "{:.2f}".format(y2), ':')
            else:
                print('click on y = ', "{0:.3E}".format(y2), ':')
        elif count == 3:
            print('Anchor points extracted, now extract data points')
    
        if clicks:
            if count == 4:
                t1 = TimeNow()
            # Get the x and y coordinates of the click
            x_click, y_click = clicks[0]
            x_pixel.append(x_click)
            y_pixel.append(y_click)
            # print(f"Clicked at ({x_click}, {y_click})")
            count = count + 1
        
        cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
        
        if count > 30:
            print('Collected a total of ', count, 'points, to end this session, move your cursor to x_pix < ', x_pix_min)
        if x_click < x_pix_min:
            if count < 4:
                raise Exception('clicked on end point before any useful data points were extracted, try reducing x_pix_min')
            else:
                proceed = 0
    t2 = TimeNow()
    print('Read_Curve_GUI: finished in ', "{:.1f}".format(t2 - t1), ' seconds, extracted ', count - 4, ' data points')
    def get_real_coordinates(pix, axis):
        if axis == 'x':
            if LogX:
                f1 = np.log10(x1)
                f2 = np.log10(x2)
            else:
                f1 = x1
                f2 = x2
            v1 = pix[0]
            v2 = pix[1]
        else:
            if LogY:
                f1 = np.log10(y1)
                f2 = np.log10(y2)
            else:
                f1 = y1
                f2 = y2
            v1 = pix[2]
            v2 = pix[3]
        v = pix[4 : len(pix)]
        f = (f2 - f1)*(v - v1)/(v2 - v1) + f1
        return f
    
    x_pixel = np.array(x_pixel)
    y_pixel = np.array(y_pixel)
    
    # the last element should be ignored, because it's the point below x_pix_min and thereby marking end of program
    x_pixel = np.delete(x_pixel,count-1)
    y_pixel = np.delete(y_pixel,count-1)
    
    x = get_real_coordinates(pix = x_pixel, axis = 'x')
    y = get_real_coordinates(pix = y_pixel, axis = 'y')
    if LogX: x = 10**x
    if LogY: y = 10**y
    
    if show_plot:
        FontSize = 15
        LineWidth = 1.5
        plt.rcParams.update({'font.family':'Times'})
        plt.rcParams['text.usetex'] = True
        fig, axs = plt.subplots(1, 2, sharex = False, sharey = False)
        fig.set_size_inches(8, 4)

        image = Image.open(IMG_File)
        axs[0].imshow(image)
        axs[0].axis('off')  # Remove axis ticks and labels
        # cursor = Cross_hair_Cursor(ax)
        # fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
        axs[0].set_title('Input Image',fontsize=FontSize)

        axs[1].plot(x, y, 'k', linewidth = LineWidth)
        axs[1].set_title('Is this your data?',fontsize=FontSize)
        axs[1].set_xlabel('$x$',fontsize=FontSize,fontname='Times New Roman')
        axs[1].set_ylabel('$y$',fontsize=FontSize,fontname='Times New Roman')
        #axs[1].set_xlim(x.min(), x.max())
        #axs[1].set_ylim(y1, y2)

        if LogX:
            axs[1].set_xscale('log')
        if LogY:
            axs[1].set_yscale('log')
        plt.xticks(size=FontSize)
        plt.yticks(size=FontSize)
        plt.tight_layout()
        try:
            plt.savefig('/tmp/tmp_Read_Curve_GUI_comparison_plot.png', dpi=300)
            print('Read_Curve_GUI: comparison plot saved to:')
            print('/tmp/tmp_Read_Curve_GUI_comparison_plot.png')
        except:
            warnings.warn('Read_Curve_GUI: Unable to save comparison plot')
    np.savez(OutputFile, x = x, y = y)
    return x,y
    
def show_status_info(idx, n, nx, name = ' ', show_status = 0):
    '''
    Show status in a loop
    idx : current index
    n : total loop number
    nx : How many times to show report
    name : additional note, e.g. who is calling this report
    show_status : whether to show report
    '''
    if show_status:
        dn = int(np.floor(n/nx))
        dn = max(1, dn)
        if nx < 100:
            fmt = "{:.2f}"
        elif nx < 1000:
            fmt = "{:.3f}"
        elif nx < 10000:
            fmt = "{:.4f}"
        else:
            fmt = "{:0.6f}"
        if idx%dn == 0:
            print(name, 'status: ',  fmt.format(idx/n))    

def Resize_chains(
        Old_Root = '/Users/cangtao/cloud/GitHub/Radio_Excess_EDGES/data/29_MCG_HII_50/29_MCG_HII_50_', 
        New_Root = '/tmp/tmp_chains_resized',
        frac = 1):
    '''
    Only keep a fraction of frac chains, new chains are saved to an alternative location,
    this would be useful if you want to make a evo getdist plot. maybe there is a smarter way to do this...
    '''
    Chain = np.loadtxt(Old_Root + '.txt')
    Chain_Len = len(Chain)
    if frac > 0.9999:
        NewChain = Chain
    else:
        idx = int(np.round(Chain_Len * frac))
        NewChain = Chain[:idx]
    
    New_Chain_File = New_Root + '.txt'
    np.savetxt(fname = New_Chain_File, X = NewChain, fmt = '%.8E', delimiter = '    ')
    
    # Get Param Names and Ranges
    NF0 = Old_Root + '.paramnames'
    NF1 = New_Root + '.paramnames'
    shutil.copy(NF0, NF1)
    
    RF0 = Old_Root + '.ranges'
    RF1 = New_Root + '.ranges'
    shutil.copy(RF0, RF1)

def Find_dif(x1, x2, method=0):
    nx = len(x1)
    Small = 1e-200

    r = 0
    for idx in np.arange(0, nx):
        x1_ = x1[idx]
        x2_ = x2[idx]
        
        if method == 0:
            top = max(x1_, x2_)
            low = min(x1_, x2_)
        else:
            top = min(x1_, x2_)
            low = max(x1_, x2_)
        if abs(low) < Small:
            dif = 0
        else:
            dif = abs(1-top/low)
        r = r + dif
    r = r/nx
    
    return r

def Get_PDF(x = np.linspace(0, 100, 1000), nx = 50):
    '''
    Get probability distribution of array x, divided into nx bins
    '''
    def count_elements(xL, xR):
        dx = xR - xL
        x1 = x[x>xL]
        x2 = x1[x1<=xR]
        r = len(x2)/dx
        return r
    x_ax = np.linspace(np.min(x), np.max(x), nx+1)
    pdf = 0*x_ax
    for idx in np.arange(0, nx):
        xL = x_ax[idx]
        xR = x_ax[idx+1]
        pdf[idx] = count_elements(xL, xR)
    
    # Normalize
    pdf = pdf[0:nx] / np.trapz(x = x_ax, y = pdf)
    x_ax = x_ax[0:nx]
    return x_ax, pdf

def Get_PDF_v2(x=np.linspace(0, 1, 100), nx = 50):
    '''
    Get probability distributions of array x, divided into nx bins
    '''
    # First get CDF
    nx_ = len(x)
    x_ax = np.sort(x)
    CDF = np.zeros(nx_)
    for idx, x_ in enumerate(x_ax):
        frac = len(x_ax[x_ax<=x_])/nx_
        CDF[idx] = frac
        # detect duplicate element
        Is_Scalar(np.argmin(np.abs(x_ax-x_)),2)
        # CDF should be increasing
        if idx > 0:
            dif = CDF[idx] - CDF[idx-1]
            if not (dif > 0):
                raise Exception('Why is CDF not increasing?')
    # Now pdf
    def PDF_Kernel():
        # Cut the BS and ditch 1% to get good resolution at high density regions
        # don't use dydx unless you want spikes
        xL = np.interp(x = 1-0.995, xp = CDF, fp = x_ax)
        xR = np.interp(x = 0.995, xp = CDF, fp = x_ax)
        x_pdf = np.linspace(xL, xR, nx)
        pdf = np.zeros(nx-1)
        for idx in np.arange(0, nx-1):
            x1 = x_pdf[idx]
            x2 = x_pdf[idx+1]
            x_ = x_ax[x_ax>=x1]
            x_ = x_[x_<x2]
            pdf[idx] = len(x_)
        x_pdf = x_pdf[0:nx-1]
        pdf = pdf/np.trapz(x=x_pdf,y=pdf)
        return x_pdf, pdf
    x_pdf, pdf = PDF_Kernel()

    # 1&2 sigma, note that these are one-sided
    x68_low = np.interp(x = 1-0.68, xp = CDF, fp = x_ax)
    x68_top = np.interp(x = 0.68, xp = CDF, fp = x_ax)
    x95_low = np.interp(x = 1-0.95, xp = CDF, fp = x_ax)
    x95_top = np.interp(x = 0.95, xp = CDF, fp = x_ax)
    
    r = {
        'x_cdf':x_ax,
        'cdf' : CDF,
        'x_pdf' : x_pdf,
        'pdf' : pdf,
        'low_68' : x68_low,
        'top_68' : x68_top,
        'low_95' : x95_low,
        'top_95' : x95_top}
    return r

def Get_Chain_stat(FileRoot, name):
    '''
    Do Getdist_Marg_Stat things from chain
    '''
    Chain = np.loadtxt(FileRoot+'.txt')
    names = np.loadtxt(FileRoot+'.paramnames', dtype=str, usecols = (0,))
    # I donno why np.where doesn't work
    for idx, name_ in enumerate(names):
        if name_ == name:
            sample_idx = idx
    sample = Chain[:, 2+sample_idx]
    r = Get_PDF_v2(x=sample)
    return r

def Set_Uniform_Multinest_Prior(theta, infos):
    '''
    Set uniform priors for multinest
    '''
    N = len(theta)
    params = []
    for idx in np.arange(0, N):
        p = theta[idx]
        info = infos[idx]
        p = p * (info['max'] - info['min']) + info['min']
        params.append(p)
    return params

def Get_Numerical_Passwd(n):
    '''
    Get a random passwd key string made of random numbers
    n : passwd length
    '''
    for idx in np.arange(0, n):
        rd = str(np.random.randint(0, 10))
        if idx==0:
            r = rd
        else:
            r = r + rd
    return r

def Read_MultiNest_BestFit(
        Root = '/Users/cangtao/FileVault/Projects/Radio_Excess_EDGES/data/276_FG7_Tcal_LogST/276_FG7_Tcal_LogST_',
        n_derived = 0):
    '''
    Read best-fit params from MultiNest stat file
    '''
    warnings.warn('This function will soon be deappreciated, use Read_MultiNest_Stats hereafter')
    def count_lines(DataFile):
        # Check how many lines a file has
        with open(DataFile, 'r') as file:
            line_count = 0
            for line in file:
                line_count += 1
        return line_count
    
    ChainFile = Root+'.txt'
    StatFile = Root+'stats.dat'
    Chain = np.loadtxt(ChainFile)
    ChainShape = np.shape(Chain)
    if len(ChainShape) < 2:
        # Highly unlikely but chain may have only 1 sample
        dim = ChainShape[0] - 2 - n_derived
    else:
        dim = Chain.shape[1]-2 - n_derived
    StatLen = count_lines(StatFile)
    skipped_line = StatLen - dim
    Stats = np.loadtxt(StatFile, skiprows = skipped_line)
    
    # Check that DIMs are expected
    DIMS = Stats[:,0]
    DIMS_Expected = np.linspace(1, dim, dim)
    dif = np.sum(np.abs(DIMS_Expected - DIMS))
    if dif > 1e-20:
        raise Exception('Unexpected dimension for loaded stats')
    MAP_active = Stats[:,1] # MAP for actively fit params
    if n_derived == 0:
        return MAP_active
    # We have derived params, find best-match for MAP and their derived params
    ChainShape = np.shape(Chain)
    dist = np.zeros(ChainShape[0])
    for idx in np.arange(0, ChainShape[0]):
        dist[idx]=0
        for pid, pmap in enumerate(MAP_active):
            dist[idx] = dist[idx] + (pmap - Chain[idx,2+pid])**2
    idx_min = np.argmin(dist)
    MAP_All = Chain[idx_min, 2:ChainShape[1]]
    return MAP_All

def Print_UntraFast_info(result, LogPath):
    LnZ_DataFile = LogPath+'Bayesian_Evidence.dat'
    str1 = 'LnZ        LnZ_Err'
    str2 = "{:.4f}".format(result['logzerr']) + '       '+"{:.4f}".format(result['logz'])
    cmd1 = 'echo ' + str1 + '>>' +LnZ_DataFile
    cmd2 = 'echo ' + str2 + '>>' +LnZ_DataFile
    os.system(cmd1)
    os.system(cmd2)

def Get_2Side_Limits(x, CL=0.95):
    x2 = np.sort(x)
    n = len(x)
    idx = int(len(x2) * (1-CL)/2)
    xL = x2[idx]
    xR = x2[n - idx]
    return xL, xR

def Read_MultiNest_Stats(
        Root = '/Users/cangtao/FileVault/Projects/Radio_Excess_EDGES/data/276_FG7_Tcal_LogST/276_FG7_Tcal_LogST_'):
    '''
    Read MultiNest stats
    '''
    SwapFile = Root+'tmp_swap_stat.dat'
    # Find dimension
    Chain = np.loadtxt(Root+'.txt')
    ChainShape = np.shape(Chain)
    dim = Chain.shape[1]-2 if len(ChainShape) >= 2 else ChainShape[0]-2 # From chain, possibly having only 1 sample
    # Check dimension
    StatFile = Root+'stats.dat'
    if not os.path.exists(StatFile):
        warnings.warn('MultiNest stat file not found, likely inputs are derived chains. I am returning only MAP')
        C2 = Chain[:, 1]
        idx_best = np.argmin(C2)
        MAP = Chain[idx_best, 2:]
        r = {'MAP' : MAP}
        return r
    with open(StatFile, 'r') as file:
        lines = file.readlines()
    StatLen = len(lines)
    StatLen_fixed = 10 # For texts
    dim_stat = int((StatLen - StatLen_fixed)/3)
    if dim != dim_stat: raise Exception('Unexpected dimension or stat file format')    
    
    # Do dim tests for upcomming modules
    def Check_dim(dim_input):
        dim_0 = np.arange(1, dim+1)
        dif = np.sum(np.abs(dim_0 - dim_input))
        if dif > 1E-20: raise Exception('Unexpected dimension')

    # Mean and sigma
    # want: [4, 4 + ndim]
    head = 4 # lines to ignore
    new_lines  = lines[head:head+dim]
    with open(SwapFile, 'w') as file:
        file.writelines(new_lines)
    stat = np.loadtxt(SwapFile)
    Check_dim(stat[:,0])
    stat_shape = np.shape(stat)
    if stat_shape[0] != dim or stat_shape[1] !=3: raise Exception('Unexpected file format')
    mean = stat[:, 1]
    sigma = stat[:, 2]
    
    # Maximum Likelihood
    head = 4 + dim + 3
    new_lines  = lines[head:head+dim]
    with open(SwapFile, 'w') as file:
        file.writelines(new_lines)
    stat = np.loadtxt(SwapFile)
    Check_dim(stat[:,0])
    MaxLike = stat[:,1]
    
    # Maximum a Prior
    new_lines  = lines[StatLen-dim:StatLen]
    with open(SwapFile, 'w') as file:
        file.writelines(new_lines)
    stat = np.loadtxt(SwapFile)
    Check_dim(stat[:,0])
    MAP = stat[:,1]
    
    # LnZ
    s0 = lines[0]
    # print(s0)
    # s1 = s0.replace('Nested Sampling Global Log-Evidence           :   ', 'LnZ =')
    # s2 = s1.replace(' +/-   ', ', LnZe =')
    s1 = s0.replace('Nested Sampling Global Log-Evidence           :   ', 'nan ')
    s2 = s1.replace(' +/-   ', ' nan    nan  ') # Easy format for numpy
    file = open(SwapFile, 'w')
    print(s2, file=file)
    file.close()
    LnZ_stats = np.loadtxt(SwapFile)
    LnZ = LnZ_stats[1]
    LnZ_Error = LnZ_stats[-1]
    
    r = {'mean': mean, 
         'sigma': sigma,
         'MaxLike': MaxLike,
         'MAP': MAP,
         'LnZ': LnZ,
         'LnZ_Error': LnZ_Error}
    os.remove(SwapFile)
    return r

def h5disp(filename, show_att = 1):
    '''
    Python equivalent of MatLab h5disp
    '''
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
            count = 0
            if show_att:
                for k in obj.attrs:
                    if count == 0: print('---- Attributes ----')
                    count += 1
                    print('   ' + k + ' :', obj.attrs[k])
        elif isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}\n    Shape: {obj.shape}\n    Data Type: {obj.dtype}")
    hdf = h5py.File(filename, 'r')
    hdf.visititems(print_structure)

def count_stats_1D(x):
    '''
    Get 95% C.I. lower and upper limits + mean
    '''
    x1 = np.sort(x)
    x_mean = np.mean(x)
    nx = len(x)
    
    # 95% C.I.
    dn = int(nx*0.025)
    x2 = x1[dn:nx-dn]
    x_L95 = np.min(x2)
    x_T95 = np.max(x2)
    
    # 68% C.I.
    dn = int(nx*0.16)
    x2 = x1[dn:nx-dn]
    x_L68 = np.min(x2)
    x_T68 = np.max(x2)

    r = {'mean': x_mean, 'L95': x_L95, 'T95': x_T95, 'L68': x_L68, 'T68': x_T68}
    return r

def count_stats_2D(x):
    '''
    Get 95% C.I. lower and upper limits + mean, x shape = [sample_size, other_axis]
    '''
    x_shape = np.shape(x)
    ns = x_shape[0]
    nx = x_shape[1]
    mean = np.zeros(nx)
    L95 = np.zeros(nx)
    T95 = np.zeros(nx)
    L68 = np.zeros(nx)
    T68 = np.zeros(nx)

    for idx in np.arange(0, nx):
        stat = count_stats_1D(x[:, idx])
        mean[idx] = stat['mean']
        L95[idx] = stat['L95']
        T95[idx] = stat['T95']
        L68[idx] = stat['L68']
        T68[idx] = stat['T68']
        
    r = {'mean': mean, 'L95': L95, 'T95': T95, 'L68': L68, 'T68': T68}
    return r
