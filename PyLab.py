''' Some useful functions
- Read_Curve
- print_mcmc_info
- Getdist_Marg_Stat
- mcmc_derived_stat
- Find_Index
- Interp_3D
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
- LCDM_HyRec
- SaySomething
'''

import numpy as np
import shutil, getdist, os, time
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from joblib import Parallel, delayed
from PIL import Image
from scipy import interpolate
import matplotlib.pyplot as plt

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
    
def print_mcmc_info(FileRoot, info):
    '''Print parameter names and ranges for getdist, useful for pymultinest post-processing
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
                      Param_Name = 'fR'):
  '''
  Get the marginalised stat of a param
  '''
  s = getdist.mcsamples.loadMCSamples(Root)
  stats = s.getMargeStats()
  lim = stats.parWithName(Param_Name).limits
  l0=lim[0].lower
  l1=lim[1].lower
  u0=lim[0].upper
  u1=lim[1].upper
  mean = s.mean(Param_Name)
  r = {'mean':mean, 'low_68':l0, 'low_95':l1, 'upper_68':u0, 'upper_95':u1}
  return r

def mcmc_derived_stat(
        model_function = lambda x: np.sum(x),
        FileRoot = '/Users/cangtao/cloud/GitHub/Radio_Excess_EDGES/data/5_UVLF/5_UVLF_',
        cache_loc='/tmp/',
        cleanup = True,
        print_status = False,
        ncpu = 1):
    '''
    Get statistics for derived params, currently not compatible with mpi CosmoMC
    example can be found in examples/example_mcmc_derived_stat.py
    ----inputs----
    model_function : a function which uses original params to get derived param
    FileRoot : Original mcmc chain file root
    cache_loc : location to operate new chains
    cleanup : whether to clean up files produced during the call
    ----outputs----
    an array of form:
    Result[0][:] = mean
    Result[1][:] = low_68
    Result[2][:] = upper_68
    Result[3][:] = low_95
    Result[4][:] = upper_95
    '''
    
    ChainFile = FileRoot + '.txt'
    NewRoot = cache_loc + '/tmp_mcmc_post_processing'
    
    # ----Prepare new chains----
    chain = np.loadtxt(ChainFile)
    ChainSize = np.shape(chain)
    Sample_Size = ChainSize[0]
    Sample_Length = ChainSize[0]
    Sample_Width = ChainSize[1]
    nparam = ChainSize[1] - 2
    
    # update name and range files
    # nothing to be done for range file except for copying it to new location
    RangeFile = FileRoot + '.ranges'
    New_RangeFile = NewRoot + '.ranges'
    shutil.copy(RangeFile, New_RangeFile)
    # Get new name file
    NameFile = FileRoot + '.paramnames'
    New_NameFile = NewRoot + '.paramnames'
    shutil.copy(NameFile, New_NameFile)
    nf=open(New_NameFile,'a')
    print('p_derived*    p_{\mathrm{derived}}',file = nf)
    nf.close()

    # Determine function type
    test_result = model_function(chain[0][2:Sample_Width])
    result_dimension = np.shape(test_result)
    if result_dimension == ():
        # This is too much work, I am not gonna work on scalar functions
        n_derived = 1
        raise Exception('This module is not yet compatible with scalar functions')
    else:
        n_derived = result_dimension[0]
    
    # Currently I can think of 5 statistics I want to save, creating size 20 for redundency
    Result = np.empty((20,n_derived))
    
    if ncpu == 1:
        Param_Array = np.empty((Sample_Length, n_derived))
        # Get param array
        for sample_id in np.arange(0, Sample_Length):
            param_list = chain[sample_id][2:Sample_Width]
            if sample_id == 0:
                # calculated for this sample_id, let's not waste it
                derived_param = test_result
            else:
                # It's up to you to ensure that model_function gets correct result from param_list
                derived_param = model_function(param_list)
            for param_id in np.arange(0, n_derived):
                Param_Array[sample_id][param_id] = derived_param[param_id]
            if print_status:
                print('Status from mcmc_derived_stat: ', sample_id/Sample_Length)
    else:
        param_list = chain[:,2:Sample_Width]
        #print(np.shape(param_list))
        #print(param_list[0])
        Param_Array = Parallel(n_jobs=ncpu)(delayed(model_function)(x) for x in param_list)
        pass

    
    # Get new chain
    NewChain = np.empty((Sample_Length, Sample_Width + 1))
    New_ChainFile = NewRoot + '.txt'
    
    for param_id in np.arange(0, n_derived):
        # Work individual param
        for sample_id in np.arange(0, Sample_Length):
            for id in np.arange(0, Sample_Width):
                NewChain[sample_id][id] = chain[sample_id][id]# copy chain
            NewChain[sample_id][Sample_Width] = Param_Array[sample_id][param_id] # appending derived param
        # NewChain ready
        # ----getdist----
        np.savetxt(New_ChainFile, NewChain, fmt='%.8E', delimiter='  ')
        r = Getdist_Marg_Stat(NewRoot, 'p_derived')
        Result[0][param_id] = r['mean']
        Result[1][param_id] = r['low_68']
        Result[2][param_id] = r['upper_68']
        Result[3][param_id] = r['low_95']
        Result[4][param_id] = r['upper_95']
    # clean-up
    if cleanup:
        os.remove(New_ChainFile)
        os.remove(New_RangeFile)
        os.remove(New_NameFile)
    return Result

'''
def Find_Index(x = 0.5, x_axis = np.linspace(0, 10, 11)):
    # Find left neibouring index
    # limit length for current method, 
    # use iterative in next revision (more efficient)
    id_max = 100000
    if len(x_axis) > id_max:
        raise Exception('x_axis too large')
    if x_axis[0] > x_axis[-1]:
        raise Exception('x_axis must be increasing')
    if x < x_axis[0] or x > x_axis[-1]:
        raise Exception('x is not in range')
    idx = id_max
    for id in np.arange(0, len(x_axis)):
        if x_axis[id] <= x and x < x_axis[id + 1]:
            idx = id
    if idx == id_max:
        raise Exception('index not found')
    return idx
'''


def Find_Index(x = 0.5, x_axis = np.linspace(0, 10, 11)):

    '''
    Find left neibouring index
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


def Interp_3D(Tab, x_axis, y_axis, z_axis, x, y, z, Linear_Axis = True):
    '''
    Interpolate a 3D array
    Tab: [xid, yid, zid]
    '''
    
    # do x-axis first
    id1 = Find_Index(x, x_axis)
    id2 = id1 + 1
    x1 = x_axis[id1]
    x2 = x_axis[id2]
    f1 = Tab[id1,:,:]
    f2 = Tab[id2,:,:]
    f = (f2 - f1)*(x - x1)/(x2 - x1) + f1
    
    # next the y-axis
    id1 = Find_Index(y, y_axis)
    id2 = id1 + 1
    y1 = y_axis[id1]
    y2 = y_axis[id2]
    g1 = f[id1,:]
    g2 = f[id2,:]
    g = (g2 - g1)*(y - y1)/(y2 - y1) + g1

    # finally the z axis
    id1 = Find_Index(z, z_axis)
    id2 = id1 + 1
    z1 = z_axis[id1]
    z2 = z_axis[id2]
    r1 = g[id1]
    r2 = g[id2]
    r = (r2 - r1)*(z - z1)/(z2 - z1) + r1

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

def Hubble(z=0):
  OmM = 0.3111
  OmL = 0.6888
  h = 0.6766
  H0 = h*3.240755744239557E-18
  zp3 = pow(1+z, 3)
  r = H0 * np.sqrt(OmL + OmM * zp3)
  return r

def Find_Negative_Element(x = np.array([1,2,-2,3]), Small = 1e-200, model = 1):
    r = 0
    Xmin = np.min(x)
    if Xmin < -Small:
        r = 1
        if model == 1:
            raise Exception('Found negative element')
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
        Use_Booster = 0,
        Use_log_x = 1,
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

    Boost = 3
    convergence_fraction = 0.2

    Small = 1e-200
    if MinX >= MaxX:
        raise Exception('Setting error, MinX is larger than MaxX!')
    
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
                # could be that f_vec is flat on the left, I am gonna allow that for first 10 iterations
                if count > 10:
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
        
    lx_left = lx1

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
        
    lx_right = lx2

    # You can do a booster run now if you like, but that's too much work for now

    return x_vec, f_vec
    
'''
def Solve(
        F = lambda x:x**3 - 3,
        Xmin = 0,
        Xmax = 10,
        Precision = 1e-3,
        show_status = 0
        ):
    # Find solution for F(x) = 0
    Small = 1e-200
    CountMax = 10000
    x1 = Xmin
    x2 = Xmax
    x_ = x1 + (x2 - x1)/2
    
    Proceed = 1
    count = 0
    
    while Proceed:
        count = count + 1
        f1 = F(x1)
        f2 = F(x2)
        f_ = F(x_)
        if f1*f2 > 0:
            Proceed = 0
            raise Exception('No solution found in this range')
        
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
        
        if f1*f_ < 0:
            x1 = x1
            x2 = x_
        else:
            x1 = x_
            x2 = x2
        x_ = x1 + (x2 - x1)/2
        dif = np.abs(x2-x1)
        
        # Check convergence
        if x1 < Small:
            if dif < Precision:
                Proceed = 0
                r = x_
        else:
            dif = dif/x1
            if dif < Precision:
                Proceed = 0
                r = x_
            
        if count > CountMax:
            Proceed = 0
            r = 1e-200/1e-200
            print('Crash iminent, debug info:')
            print('x1 = ', x1, 'x_ = ', x_, 'x2 = ', x2)
            print('f1 = ', f1, 'f_ = ', f_, 'f2 = ', f2)
            print('count = ', count)
            
            raise Exception('Solution not found within permitted counts')
        
        if show_status:
            print('count = ', count, ', dif = ', dif)
            
    return r
'''


def Solve(
        F = lambda x:x**3 - 3,
        Xmin = 0,
        Xmax = 10,
        Precision = 1e-3,
        show_status = 0
        ):
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
        # f1 = F(x1)
        # f2 = F(x2)
        f_ = F(x_)
        if f1*f2 > 0:
            Proceed = 0
            raise Exception('No solution found in this range')
        
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
            print('count = ', count, ', dif = ', dif)
            
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
    if Use_Log_Z and np.min(Tab) > Small:
        Fxy = np.log10(Tab)
        Fxy_is_log = 1
    else:
        Fxy = Tab
        Fxy_is_log = 0

    # do x-axis first
    id1 = Find_Index(x, x_vec)
    id2 = id1 + 1
    x1 = x_vec[id1]
    x2 = x_vec[id2]
    f1 = Fxy[id1, :]
    f2 = Fxy[id2, :]
    Fy = (f2 - f1)*(x - x1)/(x2 - x1) + f1
    
    # now do y-axis
    id1 = Find_Index(y, y_vec)
    id2 = id1 + 1
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
        Use_log = False
    ):
    '''
    Get dy/dx from array
    ------inputs----
    x : x array
    y : y array
    Use_log : do derivative in log space, might give better precision if data is linear-log
    '''
    if Use_log:
        v = np.log(x)
        f = np.log(y)
    else:
        v = x
        f = y
    nx = len(x)
    r = np.linspace(0,1,nx)

    for idx in np.arange(0, nx-1):
        dv = v[idx+1] - v[idx]
        df = f[idx+1] - f[idx]
        r[idx] = df/dv
    
    r[nx-1] = r[nx-2]
    if Use_log:
        # r is now dlny/dlnx, convert now
        r = r*x/y
    return r

def SaySomething(File = '/Users/cangtao/Desktop/tmp.txt'):
    '''
    Say something to the File, useful for MPI status monitor
    '''
    cmd = 'echo ---- >> ' + File
    os.system(cmd)
