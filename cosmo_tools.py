'''
Some useful python functions fro 21cmFAST related calculations, to be able to use this from any locations, add this to your bash_profile:
export PYTHONPATH=Current_Path:${PYTHONPATH}
Functions:
    compute_power
    PowerSpectra_Coeval_Kernel
    HMF
    MUV2Mh
    SFRD
    UVLF
    Radio_Temp_Astro
    Get_P21c_Coeval_cache_PS : get PS from p21c coeval cache
'''

import numpy as np
import h5py, os, warnings, tqdm

try:
    from powerbox.tools import get_power
except:
    pass
try: 
    import tools21cm as t21c 
except: 
    pass
try: 
    import py21cmfast as p21c
except:
    pass
warnings.warn('from PyLab import * has been replaced with import PyLab as PL, many of funcitons in this module needs to be updated')
import PyLab as PL
try:
    from hmf import MassFunction
except:
    pass
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from Useful_Numbers import Cosmology as cosmo

class P21c_PS_overflow(Exception):
    def __init__(self, z, zmin, zmax):
        print('Exception in cosmo_tools.Get_p21c_PSk_Kernel: Cannot find cube for redshift', z, ', lc range = ', [zmin, zmax])

# Load some Interpolation Tables
def Load_Interp_Tables():
    pythonpaths = os.environ.get('PYTHONPATH').split(os.pathsep)
    PyLab_path = [path for path in pythonpaths if 'PyLab' in path]
    if len(PyLab_path)==0:
        raise Exception('Found no PyLab path')
    PyLab_path = PyLab_path[0]
    # 1 : Load HMG, author: PyLab/data/HMF_Table.py
    HMF_Table_File = PyLab_path + 'data/HMF_Interp_Table.npz'
    HMF_Table = np.load(HMF_Table_File)
    # HMF_Table contains the following fields:
    # dndm, z, m

    # 2 : Load cosmic age, author: PyLab/data/cosmic_age_table.py
    Age_Table_File = PyLab_path + 'data/cosmic_age_table.npz'
    Cosmic_Age_Table = np.load(Age_Table_File)
    # Age_Table contains the following fields:
    # LgZp_axis, LgT_axis

    Interp_Table = {
        'HMF_Table': HMF_Table,
        'Cosmic_Age_Table' : Cosmic_Age_Table
        }
    return Interp_Table

# Halo_Interp_Table = Load_Interp_Tables()
Interp_Table = Load_Interp_Tables()

def PowerSpectra_Coeval_Kernel(
        Field = 0, 
        nk = 50,
        Use_LogK = True, 
        DataFile = '/Volumes/18810925771/21cmFAST-data/cache/BrightnessTemp_ff5ac1057fff286fb047bfeb950cbb84_r64301115523.h5'):
    '''
    Calculate Power Spectra for LightCone object
    ---- Inputs ----
        Field: Choose quantities
            0 - Tb
            1 - Tk
            2 - xH
            3 - xe
            4 - Tr
            5 - Boost
            6 - Density
        nk : number of k bins
        Use_LogK : Use log k-bin
        DataFile: File name for Power spectra
    '''
    FieldNames = ['BrightnessTemp/brightness_temp', #0 - Tb
                  'TsBox/Tk_box', #1 - Tk
                  'IonizedBox/xH_box', #2 - xH, Incorrect field
                  'TsBox/x_e_box', #3 - xe
                  'TsBox/Trad_box', #4 - tr
                  'TsBox/Boost_box', #5 - Boost
                  'PerturbedField/density', #6 - density
                  ]
    f = h5py.File(DataFile, 'r')
    BOX_LEN = f['user_params'].attrs['BOX_LEN']
    Box = f[FieldNames[Field]]
    Pk, k = compute_power(
        box = Box,
        length = (BOX_LEN,BOX_LEN,BOX_LEN),
        log_bins = Use_LogK,
        nk = nk,
        ignore_kperp_zero = True,
        ignore_kpar_zero = False,
        ignore_k_zero = True)
    Ps=Pk * k ** 3 / (2 * np.pi ** 2)
    # remove nan
    NanIdx = []
    NanCheck = np.isnan(Ps)
    for kid in np.arange(0, len(k)):
        if NanCheck[kid]:
            NanIdx.append(kid)
    k = np.delete(k, NanIdx)
    Ps = np.delete(Ps, NanIdx)
    f.close()
    return k, Ps
    
def compute_power(
   box,
   length,
   nk,
   log_bins = True,
   ignore_kperp_zero = True,
   ignore_kpar_zero = False,
   ignore_k_zero = True,
):
    # Determine the weighting function required from ignoring k's.
    k_weights = np.ones(box.shape, dtype=int)
    n0 = k_weights.shape[0]
    n1 = k_weights.shape[-1]

    if ignore_kperp_zero:
        k_weights[n0 // 2, n0 // 2, :] = 0
    if ignore_kpar_zero:
        k_weights[:, :, n1 // 2] = 0
    if ignore_k_zero:
        k_weights[n0 // 2, n0 // 2, n1 // 2] = 0

    res = get_power(
        box,
        boxlength=length,
        bins=nk,
        bin_ave=False,
        get_variance=False,
        log_bins=log_bins,
        k_weights=k_weights,
    )

    res = list(res)
    k = res[1]
    if log_bins:
        k = np.exp((np.log(k[1:]) + np.log(k[:-1])) / 2)
    else:
        k = (k[1:] + k[:-1]) / 2
    res[1] = k
    return res

def HMF(z=0,
        model = 1,
        Mmin = 1e2,
        Mmax = 1E18,
        nm = 100,
        POWER_SPECTRUM = 0,
        Use_Interp = False):
    '''
    An interface with hmf package
    -- inputs --
    z : z
    model : hmf model, follows 21cmFAST convention
            0 - PS
            1 - ST
    Mmin : minimum halo mass in Msun
    Mmax : maximum halo mass in Msun
    nm : number of mass points
    POWER_SPECTRUM : Transfer Function
                     0 - EH
                     1 - BBKS
    -- outputs --
    m : mh in Msun
    dndm : dn/dm in Mpc^-3 Msun^-1
    '''
    if Use_Interp:
        # This can improve speed by 18 times
    
        # check whether the data is ok

        HMF_Tab = Interp_Table['HMF_Table']
        z_axis = HMF_Tab['z']
        m_axis = HMF_Tab['m']

        model_ok = (model == 1)
        PS_ok = (POWER_SPECTRUM == 0)
        z_ok = PL.Within_Range(z, z_axis)
        m_ok = PL.Within_Range(Mmin, m_axis) and PL.Within_Range(Mmax, m_axis)
        All_Clear = model_ok and PS_ok and z_ok and m_ok

        if All_Clear:
            Tab = HMF_Tab['dndm']
            idx1 = PL.Find_Index(x = z, x_axis = z_axis)
            idx2 = idx1 + 1
            z1 = z_axis[idx1]
            z2 = z_axis[idx2]
            # Interpolate in log axis
            x  = np.log10(1+z)
            x1 = np.log10(1+z1)
            x2 = np.log10(1+z2)
            y1 = Tab[idx1,:]
            y2 = Tab[idx2,:]
            # Get dndm at z
            fz = (y2 - y1)*(x - x1)/(x2 - x1) + y1
            # Now do m axis in log
            lm1 = np.log10(Mmin)
            lm2 = np.log10(Mmax)
            m = np.logspace(lm1, lm2, nm)
            lm = np.log10(m)
            lm_axis = np.log10(m_axis)
            dndm = np.interp(x = lm, xp = lm_axis, fp = fz)
            return m, dndm

    h = 0.6766
    hmf_models = ["PS", "SMT"]
    transfer_models = ["EH", "BBKS"]
    lm1 = np.log10(h*Mmin)
    lm2 = np.log10(h*Mmax)
    dlm = (lm2 - lm1)/(nm - 1)

    r = MassFunction(
        z = z,
        Mmin = lm1,
        Mmax = lm2,
        dlog10m = dlm,
        hmf_model = hmf_models[model],
        transfer_model = transfer_models[POWER_SPECTRUM],
    )
    '''
    r.update(
        mdef_model  = "FOF"
    )
    '''
    m0 = r.m/h
    dndm0 = r.dndm*pow(h,4)
    if len(m0) != nm:
        m = np.logspace(np.log10(Mmin), np.log10(Mmax), nm)
        # dndm = spline(m0, dndm0)(m)
        dndm = np.interp(x = m, xp = m0, fp = dndm0)
    
        dndm[0] = dndm0[0]
        dndm[-1] = dndm0[-1]
    else:
        m, dndm = m0, dndm0
    return m,dndm
  
def MUV2Mh(MUV=-10,z=6, t_STAR = 0.5, ALPHA_STAR = 0.5, log10_f10 = -1.301):
    '''Get halo mass Mh for UV magnitude MUV, unit: msun'''
    OmM = 0.3111
    OmB = 0.04897468161
    yr = 31557600
    f10 = pow(10, log10_f10)
    # UV luminosity, in erg/s/Hz
    LUV = pow(10, 0.4 * (51.63 - MUV))
    # SFR, in Msun/s
    SFR = 1.15E-28 * LUV / yr
    H = PL.Hubble(z)
    m1 = t_STAR * OmM * pow(10, 10 * ALPHA_STAR) * SFR / (H * f10 * OmB)
    r = pow(m1, 1/(ALPHA_STAR+1))
    return r

def SFRD(
    z_ax = np.linspace(0, 10, 50),
    t_STAR = 0.5,
    ALPHA_STAR = 0.5,
    Lf10 = -1.3010,
    LMturn = 8.699,
    hmf_model = 1,
    Transfer_model = 0,
    model = 1,
    nm = 10000,
    Tvir = 1E4,
    mmin = 1E7,
    fcoll_model = 0,
    show_status=0,
    Use_Interp = 0):
    '''
    Get SFRD in Msun/yr/Mpc^3
    default settings reproduces Park 18 results, see 1809.08995
    ---- inputs ----
    model : SFRD model
        0 - use Fcoll as in original 21cmFAST
        1 - use mturn as in Park18
    '''
    if nm<200:
        print('Warning from SFRD: nm<200, precision might be compromised, recommended: nm = 10000')
    # Settings and conversions
    m1 = 1e2
    m2 = 1e22
    # nm = 10000
    OmM = 0.3111
    OmB = 0.04897468161
    Mturn = pow(10, LMturn)
    f10 = pow(10, Lf10)
    OmMh2 = 0.14175010989853876
    RhoM_cmv = 2.775e11 * OmMh2# msun/Mpc^3
    year = 31557600
    nz = len(z_ax)
    r = np.zeros(nz)

    if model == 0:
        dz = 0.05
        dfcoll_dt = np.zeros(nz)
        for idx, z in enumerate(z_ax):
            if show_status:
                print(idx/nz)
            z_vec = np.array([z, z+dz])
            t = z2t(z=z_vec, unit=1, Use_interp=1)
            fcoll = fcoll_function(z=z_vec, mmin_model=fcoll_model,mmin=mmin, nm=nm, show_status=0,Tvir=Tvir,
                               Use_Interp = Use_Interp)
            dfcoll = fcoll[1] - fcoll[0]
            dt = t[1] - t[0]
            dfcoll_dt[idx] = dfcoll/dt
        dRhoStardt = dfcoll_dt * RhoM_cmv
        return dRhoStardt

    # Starting
    f_duty = np.linspace(1,100,nm)
    SFR = np.linspace(1,100,nm)
    for zid in np.arange(0,nz):
        z = z_ax[zid]
        H = PL.Hubble(z)
        mh, dndmh = HMF(z, hmf_model, m1, m2, nm, Transfer_model)
        for id in np.arange(0,nm):
            m = mh[id] # Msun
            f_duty[id] = np.exp(-Mturn/m)
            fstar = f10*pow(m/1e10, ALPHA_STAR)
            Mb = m*OmB/OmM
            Mstar = Mb*fstar
            SFR[id] = Mstar*H/t_STAR # Msun/s
            f = dndmh * f_duty * SFR
        r[zid] = np.trapz(y = f, x = mh) * year
    return r

def UVLF(M1 = -22,
        M2 = -8,
        nm = 100,
        z = 6,
        t_STAR = 0.5,
        ALPHA_STAR = 0.5,
        log10_f10 = -1.3010,
        log10_Mturn = 8.699,
        hmf_model = 1,
        Transfer_model = 0):
    # Get Settings
    OmM = 0.3111
    OmB = 0.04897
    M_TURN = 10**log10_Mturn
    f10 = 10**log10_f10
    H = PL.Hubble(z)
    mh2 = MUV2Mh(M1, z, t_STAR, ALPHA_STAR, log10_f10)
    mh1 = MUV2Mh(M2, z, t_STAR, ALPHA_STAR, log10_f10)
    year = 31557600
  
    # Determine hmf sample size, 50 per order of magnitude
    nm_sample = np.rint(50*np.log10(mh2/mh1)).astype(int)
    
    # Get hmf template
    m0, dndm0 = HMF(z, hmf_model, mh1, mh2, nm_sample, Transfer_model)
    m0[0] = mh1
    m0[-1] = mh2
    Prefix = H * OmB * f10 / (t_STAR * OmM * pow(10, 10*ALPHA_STAR)) # DIM: s^-1
    MUV_array = np.linspace(M1, M2, nm)
    r = np.linspace(M1, M2, nm)
    for id in np.arange(0,nm):
        MUV = MUV_array[id]
        m = MUV2Mh(MUV, z, t_STAR, ALPHA_STAR, log10_f10) # halo mass
        f_duty = np.exp(-M_TURN/m)
        dndm = spline(m0,dndm0)(m) # unit: Mpc^-3 Msun^-1
        dm_dSFR = pow(m, -ALPHA_STAR)/Prefix/(1+ALPHA_STAR) # unit : s
        dSFR_dMUV = -4.751/(pow(10, 8+0.4*MUV))/year # unit: Msun s^-1
        dm_dMUV = dm_dSFR * dSFR_dMUV # unit: Msun
        r[id] = f_duty * dndm * dm_dMUV # unit: Mpc^-3
    return MUV_array, np.abs(r)

def Radio_Temp_Astro(
    fR = 3162.27,
    aR = 0.7,
    nz = 100,
    zcut = 17,
    z_ax = np.linspace(5, 30, 100),
    SFRD_ax = np.logspace(-1, -8.64, 100)):
    '''
    Get Radio Temp for astrophysical sources
    ---- Inputs ----
    fR : fR
    aR : power-index
    nz : number of z bins in which you want Tradio, default is between min(z_ax) and max(z_ax)
    zcut : turn off radio emission below this
    z_ax : axis for z in SFRD_ax
    SFRD_ax : SFRD in Msun/yr/Mpc^3
    '''
    # some numbers
    Mpc = 3.086E22
    Msun = 1.98847E30
    Yr = 365.25 * 24 * 3600
    c = 299792458
    v0 = 0.15E9
    v21 = 1.429E9
    kB = 1.38064852E-23
    Prefix = 0.1587 * pow(c,3) * pow(v0, aR) / (8 * np.pi * kB * pow(v21, 2 + aR))
    zmin = np.min(z_ax)
    zmax = np.max(z_ax)

    z_new = np.linspace(zmin, zmax, 10000)
    SFRD_new = np.interp(z_new, z_ax, SFRD_ax)
    SFRD_SI = SFRD_new * Msun / Yr / pow(Mpc, 3)
    H = PL.Hubble(z_new)
    z = np.linspace(zmin, zmax, nz)
    T = np.zeros(nz)
    xp = np.log(1+z_new)
    F = SFRD_SI/H/(pow(1+z_new, aR))

    for idx in np.arange(0, nz):
        z_ = z[idx]
        f = fR * pow(1+z_, 3+aR) * Prefix * F
        f = f * np.heaviside(z_new - z_, 0)
        f = f * np.heaviside(z_new - zcut, 0)
        T[idx] = np.trapz(y = f, x = xp)

    return z, T

def Flat_Gaussian(
        a21 = 0.53,
        v0 = 78.3,
        t = 6.5,
        w = 20.7,
        v = np.linspace(50, 100, 130),
        ):
    '''
    Flattened Gaussian T21 profile (in K)
    default params are the ones obtained in B18
    ----inputs----
    a21 : T21 amplitude in K
    v0 : central frequency in MHz
    w : FWHM
    t : flattening factor
    '''
    b1 = np.log( - np.log((1 + np.exp(-t))/2)/t )
    b2 = 4*(v-v0)**2/w**2
    B = b1*b2
    t1 = 1 - np.exp( -t * np.exp(B) )
    t2 = 1 - np.exp( -t )
    t = -a21 * t1/t2
    return t

def HaloProfile_Kernel(
        z = 1,
        mh = 1e4,
        r = 10,
        OmB = 0.04897468161,
        ProfileType = 1):
    '''
    Get halo density profile, note NFW is for DM, not DM+baryon.
    We will assume that DM and baryon follow same distribution
    ---inputs----
    z : z
    m : halo mass in msun, note that this is for DM+Baryon
    r : distance in pc
    OmB : Omega_b
    ProfileType : choose profile type
                  0 - Matter (dm + baryon)
                  1 - dm
                  2 - baryon
                  3 - Just give r_vir
    ------outputs----
    density profile for matter, dark matter, baryon, or r_vir. density unit: msun/pc^3
    '''
    # Some settings
    OmM = 0.30964168161
    OmC = 0.260667
    OmR = 9.1e-5
    OmL = 0.69026731839
    # OmC = 0.260667
    h = 0.6766
    m = mh * OmC/OmM
    pi = 3.141592653589793
    rho_cr0 = 2.775e-7 * h**2 # critical density in msun/pc^3
    
    # Pre-requisites
    zp = 1 + z
    OmMz = OmM * zp**3 /(OmM * zp**3 + OmL)
    d = OmMz - 1
    Delta_C = 18 * pi**2 + 82 * d - 39 * d**2
    log10_c = 1.071 - 0.098 * (np.log10(m) - 12)
    c = 10**log10_c/zp # concentration, see appdx.A of Zip.et for the additional (1+z) factor
    delta_c = Delta_C * c**3 / (3 * (np.log(1 + c) - c/(1+c)))
    
    rv1 = 0.784 * (m * h/1e8)**(1/3)
    rv2 = (OmM * Delta_C / (OmMz * 18 * pi**2))**(-1/3)
    rv3 = (10/zp)/h * 1000
    r_vir = rv1 * rv2 * rv3
    
    def get_rho_dm():
        x = r/r_vir
        cx = c*x
        rho_cr = rho_cr0 * (OmL + OmM * zp**3 + OmR * zp**4)
        RhoDM = rho_cr * delta_c /(cx * (1 + cx)**2)
        RhoDM = RhoDM * np.heaviside(x, 1)
        return RhoDM
    
    RhoDM = get_rho_dm()

    if ProfileType == 3:
        return r_vir
    elif ProfileType == 1:
        return RhoDM
    else:
        raise Exception('Module not ready, you might be looking for the older version of HaloProfile_Kernel (commented out)')

def HaloProfile(
        z = 10,
        mh = 1e4,
        OmB = cosmo['OmB'],
        nr = 1000,
        map_nx = 100,
        map_precision = 1e-4,
        mass_error = 0.05
        ):
    
    Small = 1e-200
    profile_kernel = lambda x : HaloProfile_Kernel(z = z, m_h = mh, r = x, Omega_b = OmB, ProfileType = 0)
    # The mapping integration is done in lnx, so it makes sense to use [4 pi r^3 rho] as kernel
    map_kernel = lambda x : 4 * np.pi * x**3 * profile_kernel(x)
    r_vir = HaloProfile_Kernel(z = z, m_h = mh, r = 1, Omega_b = OmB, ProfileType = 3)
    lr_vir = np.log10(r_vir)
    Start = lr_vir - 1
    
    x, fx = PL.Map(
        F = map_kernel,
        Start = Start,
        Width = 1,
        MinX = -np.inf,
        MaxX = lr_vir,
        nx = map_nx,
        Precision = map_precision,
        Max_Iteration = 100,
        Use_log_x = 1
        )
    
    M = np.trapz(x = np.log(x), y = fx)
    dif = np.abs(M - mh)/M
    
    if dif > mass_error:
        # Try again with higher precision
        x, fx = PL.Map(
            F = map_kernel,
            Start = Start,
            Width = 1,
            MinX = -np.inf,
            MaxX = lr_vir,
            nx = 5 * map_nx,
            Precision = map_precision,
            Max_Iteration = 100,
            Use_log_x = 1)
        M = np.trapz(x = np.log(x), y = fx)
        dif = np.abs(M - mh)/M
        if dif > 1.5 * mass_error:
            raise Exception('Halo profile not convergent, difference = ', dif)
    
    # Ok now we have a convergent r axis
    r = x
    RhoM = fx / (4 * np.pi * r**3)
    RhoB = HaloProfile_Kernel(z = z, m_h = mh, r = r, Omega_b = OmB, ProfileType = 2)
    RhoC = RhoM - RhoB
    if nr < 0:
        result = np.empty((4, len(r)))
        result[0,:] = r
        result[1,:] = RhoM
        result[2,:] = RhoC
        result[3,:] = RhoB
        return result
    
    # Use a different axis
    # You might want to have a fixed result size, e.g. for building interpolation table
    LgR_axis = np.log10(r)
    LR1 = np.min(LgR_axis)
    LR2 = np.max(LgR_axis)
    LgR = np.linspace(LR1, LR2, nr)
    
    # try log first
    if np.min(RhoM) > Small:
        LgRhoM = np.interp(x = LgR, xp = LgR_axis, fp = np.log10(RhoM))
        Rho_M = 10**LgRhoM
    else:
        Rho_M = np.interp(x = LgR, xp = LgR_axis, fp = RhoM)
    if np.min(RhoC) > Small:
        LgRhoC = np.interp(x = LgR, xp = LgR_axis, fp = np.log10(RhoC))
        Rho_C = 10**LgRhoC
    else:
        Rho_C = np.interp(x = LgR, xp = LgR_axis, fp = RhoC)
    if np.min(RhoB) > Small:
        LgRhoB = np.interp(x = LgR, xp = LgR_axis, fp = np.log10(RhoB))
        Rho_B = 10**LgRhoB
    else:
        Rho_B = np.interp(x = LgR, xp = LgR_axis, fp = RhoB)
    R = 10**LgR
    result = np.empty((4, len(R)))
    result[0,:] = R
    result[1,:] = Rho_M
    result[2,:] = Rho_C
    result[3,:] = Rho_B
    return result

def Compute_Optical_Depth(
        z = np.linspace(0, 7.82, 2000),
        xe = 1.15*np.ones(2000),
        zmax = 100,
        xe_format = 0,
        zre_He = 3.5):
    '''
    Find optical depth, default is for a instantaneous EoR model with zre given by PLK18
    ----inputs----
    xe_format : Helium content in xe
        0 - xe = ne/nH, namely xe is total contribution from H and He
        1 - H and He are ionised to the same degree xe, like 21cmFAST, but this is likely inaccurate
    '''
    if z[0] > z[-1]:
        raise Exception('z needs to be increasing!')
    # warnings.warn('Module will soon be replaced by p21c.wrapper.compute_tau')
    
    nH = 0.1901567053460595*(1+z)**3
    sT = 0.665245854E-28
    c = 299792458
    Prefix = sT * c
    fHe = 0.08112582781456953

    if xe_format == 0:
        ne = nH*xe
    else:
        neH = nH*xe
        nHe = nH*fHe
        neHe = []
        neHe1 = xe * nHe # there are 2 electrons in He but 21cmFAST assumes H and singly ionized He share same xe
        for idx in np.arange(0, len(z)):
            if z[idx] > zre_He:
                neHe.append(neHe1[idx])
            else:
                neHe.append(2*neHe1[idx])
        neHe = np.array(neHe)
        ne = neH + neHe
    
    H = PL.Hubble(z)
    f = Prefix*ne/H/(1+z)*np.heaviside(zmax - z, 1)
    r = np.trapz(x = z, y = f)
    return r

def Switch_xe_format(xe = 1.0, format = 0):
    '''
    change between different xe format (HyRec or 21cmFAST)
    ----inputs----
    xe : ionisation fraction
    format : xe input format
        0 - HyRec, ne/nH
        1 - 21cmFAST, np/nH
    '''
    #raise Exception('This is inacurate, Helium is singly ionized in 21cmFAST')
    fHe = 0.08112582781456953 # nHe/nH for Yhe = 0.245
    if format == 0:
        r = xe/(1+2*fHe)
    else:
        r = xe * (1+2*fHe)
    return r

def fcoll_function(
        z = np.linspace(0, 10, 50),
        mmin_model = 1, 
        mmin = 1e7, 
        Tvir = 1E4, 
        Use_EoR = 1, 
        nm = 10000,
        show_status = 0, 
        Use_Interp = 1):
    '''
    Compute fcoll
    ----inputs----
    z : z
    nm : number of m bins in integration
    mmin_model : choice of mmin
        0 - fcoll for Tvir
        1 - fcoll for baryon overdensity threshold
        2 - use value provided in mmin
    '''
    OmMh2 = 0.14175010989853876
    RhoM_cmv = 2.775e11 * OmMh2# msun/Mpc^3
    
    PL.Is_Scalar(x = z, ErrorMethod = 1)
    nz = len(z)
    fcoll = np.zeros(nz)
    for idx,z_ in enumerate(z):
        if show_status:
            print(idx/nz)
        if mmin_model == 0:
            Mmin = 1E8 * (Tvir/2E4)**1.5 * ((1+z_)/10)**-1.5
        elif mmin_model == 1:
            tmp, Tk = PL.LCDM_HyRec(z = z_, Use_EoR = Use_EoR)
            Mmin = 1.3e3 * (10/(1+z_))**1.5 * Tk**1.5
        elif mmin_model == 2:
            Mmin = mmin
        m, dndm = HMF(z = z_, Mmin = Mmin, Mmax = 1e20, nm = nm, Use_Interp = Use_Interp)
        RhoHalo = np.trapz(x = np.log(m), y = m**2 * dndm)
        fcoll[idx] = RhoHalo/RhoM_cmv
    return fcoll

def lc2tau(file = '/Users/cangtao/Desktop/21cmFAST-data/EOS_2021.h5'):
    '''
    Find tau from a LightCone h5 file
    '''
    lc = p21c.LightCone.read(file)
    z = lc.node_redshifts[::-1]
    xH = lc.global_xH[::-1]
    z2 = np.linspace(3.0, z.max(), 1000)
    xH = np.interp(x = z2, xp = z, fp = xH)
    z = z2
    r = p21c.wrapper.compute_tau(redshifts = z, global_xHI = xH)
    return r

def z2t(z = 0, unit = 0, nz = 10000, max_ratio = 1.0e6, Use_interp = 0):
    
    '''
    Find cosmic age t using z
    ---- Inputs ----
    z : redshifts
    unit : unit of time
        0 - second
        1 - year
    nz : number of z integration timesteps
    max_ratio : z-integration upper limit / (1+z)
    Use_interp : use pre-computed interpolation table
    '''
    
    yr = 365.25 * 24 * 3600
    
    if Use_interp:
        LgZp = np.log10(1+z)
        lr = np.interp(x = LgZp, xp = Interp_Table['Cosmic_Age_Table']['LgZp_axis'], fp = Interp_Table['Cosmic_Age_Table']['LgT_axis'])
        t = 10**lr
        if unit == 0:
            return t
        else:
            t = t/yr    
            return t
        
    def z2x(z_):
        # convert z to lna
        a = 1/(1+z_)
        x = np.log(a)
        return x
    
    def x2z(x):
        # convert lna to z
        a = np.exp(x)
        z_ = 1/a-1
        return z_
    
    zmax = (1+z)*max_ratio-1
    x1 = z2x(zmax)
    x2 = z2x(z)
    x_vec = np.linspace(x1, x2, nz)
    z_vec = x2z(x_vec)
    H = PL.Hubble(z = z_vec)
    t = np.trapz(x = x_vec, y = 1/H)
    if unit == 0:
        return t
    else:
        t = t/yr    
        return t

def t2z(t):
    LgT = np.log10(t)
    # Interp requires x to be increasing
    LgT_axis = Interp_Table['Cosmic_Age_Table']['LgT_axis'][::-1]
    LgZp_axis = Interp_Table['Cosmic_Age_Table']['LgZp_axis'][::-1]
    Lgzp = np.interp(
        x = LgT,
        xp = LgT_axis,
        fp = LgZp_axis)
    z = 10**Lgzp - 1
    return z

def cosmic_distance(z = 1000, z_end = 0, OmM = 0.30964168161, h = 0.6766, OmR = 9.1E-5, Use_Comoving = 1, Unit = 'Mpc'):
    '''
    Distance between z and z_end
    '''
    nz = 1000
    zp_vec = np.logspace(np.log10(1+z_end), np.log10(1+z), nz)
    z_vec = zp_vec - 1
    H_vec = PL.Hubble(z = z_vec, OmM = OmM, h = h, OmR = OmR)
    c = 299792458
    Mpc = 3.086E22
    yr = 31557600
    Gyr = yr*1e9

    if Use_Comoving:
        fun = c/H_vec
    else:
        fun = c/H_vec/zp_vec
    
    l = np.trapz(x = np.log(zp_vec), y = zp_vec*fun)
    l = np.abs(l)
    
    if Unit == 'Mpc':
        l = l / Mpc
    elif Unit == 'yr':
        l = l/c/yr
    elif Unit == 'Gyr':
        l = l/c/Gyr

    return l

def d2z(d, OmM = 0.30964168161, h = 0.6766, OmR = 9.1E-5):
    '''
    For comoving distance d [Mpc]to z=0, find z
    Not very fast cause we are using solve, try using interpolation next time
    '''
    def kernel(x):
        # x = log10(1+z)
        z = 10**x - 1
        r = cosmic_distance(z = z, z_end=0, OmM = OmM, h = h, OmR = OmR, Use_Comoving=1, Unit='Mpc')
        return d - r
    x = PL.Solve(F = kernel, Xmin = 0, Xmax = 3, Precision=1e-3)
    z = 10**x - 1
    return z

def get_lc_redshifts(lc):
    '''
    只能用那一招了
    '''
    d = lc.lightcone_distances
    nz = len(d)
    z = np.zeros(nz)
    for idx in np.arange(0, nz):
        z[idx] = d2z(d[idx])
    return z

def Get_P21c_Coeval_cache_PS(path, cleanup, nk, output_file, field=0, show_status = 0):
    '''
    Get PS from cached coevals
    -- inputs --
        path : path where cached coevals are stored, must all be for the same p21c params
        cleanup : delete things inside cache path after use?
            0 - don't delete
            1 - delete ALL
            2 - keep Tb boxes and delete the rest
        nk : number of k bins
        output_file : where (npz or h5) to store PS
        Field: Choose quantities
            0 - Tb
            1 - Tk
            2 - xH
            3 - xe
            4 - Tr
            5 - Boost
            6 - Density
    '''
    FL = []
    Files = os.listdir(path)
    Heads = ['Bri', #0 - Tb
             'TsB', #1 - Tk
             'Ion', #2 - xH
             'TsB', #3 - xe
             'TsB', #4 - Tr
             'TsB', #5 - Boost
             'Per', #6 - density
             ]
    for file in Files:
        Head = file[0]+file[1]+file[2]
        Tail = file[-3]+file[-2]+file[-1]
        if Head == Heads[field] and Tail == '.h5':
            FL.append(path+file)
    z0 = []
    k2D = []
    PS0 = []
    nf = len(FL)
    #for fid, file in enumerate(FL):
    for fid in tqdm.tqdm(range(len(FL)), desc = 'Computing Coeval PS', disable = not show_status):
        file = FL[fid]
        k_, PS_ = PowerSpectra_Coeval_Kernel(Field=field,nk=nk, Use_LogK=True, DataFile=file)
        f = h5py.File(file, 'r')
        redshift = f.attrs['redshift']
        f.close()
        k2D.append(k_)
        PS0.append(PS_)
        z0.append(redshift)
    # All PS should have same k I think but let's check
    for idx in np.arange(0, nf-1):
        k1 = k2D[idx]
        k2 = k2D[idx+1]
        dif = np.sum(np.abs(k1-k2))
        if dif > 1e-50:
            raise Exception('Some coeval PS has different k')
    # Now let's sort them into same format as p21c_ps_kernel
    z_ax_0 = np.array(z0)
    k_ax = k_
    z_ax = np.sort(z_ax_0)
    nz = len(z_ax)
    nk_ = len(k_ax)
    PS = np.zeros([nz,nk_])
    for idx in np.arange(0, nz):
        z = z_ax[idx]
        idx0 = np.argmin(np.abs(z-z_ax_0))
        PS[idx, :] = PS0[idx0][:]
    # Now save results to file
    if output_file[-1] == 'z':
        # npz file
        np.savez(output_file, z=z_ax, k = k_ax, ps=PS)
    else:
        # h5 file
        f = h5py.File(output_file, 'w')
        f.create_dataset('Coeval_Power_Spectrum/z', data = z_ax)
        f.create_dataset('Coeval_Power_Spectrum/k', data = k_ax)
        f.create_dataset('Coeval_Power_Spectrum/PS', data =PS)
        f.close()
    # now cleanup
    if cleanup == 1:
        for file in Files:
            try:
                os.remove(path+file)
            except:
                pass
    elif cleanup == 2:
        for file in Files:
            Head = file[0]+file[1]+file[2]
            Tail = file[-3]+file[-2]+file[-1]
            if Tail == '.h5':
                if not Head == 'Bri':
                    os.remove(path+file)
    result = {'z': z_ax, 'k' : k_ax, 'ps' : PS}
    return result

def Get_p21c_lc_cube_redshifts(lc):
    '''
    Find z axis from p21c lc that can form cube, needed for LC PS modules
    '''
    z_ax = lc.lightcone_redshifts
    d = lc.lightcone_distances
    BOX_LEN=lc.user_params.BOX_LEN
    d1 = np.min(d) + BOX_LEN/2
    d2 = np.max(d) - BOX_LEN/2
    idx1 = np.argmin(np.abs(d1 - d))
    idx2 = np.argmin(np.abs(d2 - d))
    # Above are nearest, remove overflow
    z = z_ax[idx1+1:idx2-1]
    return z

def Get_p21c_PSk_Kernel(
    lc,
    z=10, 
    field = 0, 
    nk = 30):
    '''
    Get PS as a function of k from p21c lightcone
    result is for closest z so further interpolation is needed if we are serious
    -- Inputs --
    z : redshift
    H5File : H5 file containing the LC
    lc : p21c lightcone, cannot set default without inducing errors for machines without lc h5, but an example is
        lc = p21c.LightCone.read('/Users/cangtao/Desktop/21cmFAST-data/Radio_Paper/Fiducial.h5')
    field : lightcone quantities
        0 - Tb
        1 - Tk
        2 - xH
        3 - xe
        4 - Tr
        5 - Boost
        6 - Density
    '''
    z_ax = lc.lightcone_redshifts
    nearest_idx = np.argmin(np.abs(z-z_ax))
    z_out = z_ax[nearest_idx] # closest match
    d = lc.lightcone_distances
    xc = d[nearest_idx]
    BOX_LEN=lc.user_params.BOX_LEN
    
    # Make sure the chunk is close to a cube
    x1 = xc - BOX_LEN/2
    x2 = xc + BOX_LEN/2
    z_cube_ax = Get_p21c_lc_cube_redshifts(lc)
    zmin, zmax = np.min(z_cube_ax), np.max(z_cube_ax)

    if z_out < zmin or z_out > zmax:
        raise P21c_PS_overflow(z, zmin, zmax)
    
    if field == 0:
        box = lc.brightness_temp
    elif field == 1:
        box = lc.Tk_box
    elif field == 2:
        box = lc.xH_box
    elif field == 3:
        box = 1.0 - lc.xH_box
    elif field == 4:
        box = lc.Trad_box
    elif field == 5:
        box = lc.Boost_box
    elif field == 6:
        box = lc.density
    idx1 = np.argmin(np.abs(d-x1))
    idx2 = np.argmin(np.abs(d-x2))
    chunklen = (idx2 - idx1) * lc.cell_size
    ps0, k0 = compute_power(
            box = box[:, :, idx1:idx2],
            length = (BOX_LEN, BOX_LEN, chunklen),
            nk = nk,
            log_bins = True)
    # Remove nan
    k = []
    ps = []
    for idx, ps_ in enumerate(ps0):
        if not np.isnan(ps_):
            k.append(k0[idx])
            ps.append(ps_)
    # Dimensionless PS
    ps = np.array(ps)
    k = np.array(k)
    ps = ps * k ** 3 / (2 * np.pi ** 2)
    r = {'k':k, 'ps':ps, 'z_out':z_out}
    return r

def Get_P21c_PS(
    H5File = '/Users/cangtao/Desktop/21cmFAST-data/Radio_Paper/Template.h5', 
    nz = 30,
    k = 0.5,
    nk = 30,
    field = 0,
    output_file = '/tmp/tmp.npz',
    show_status = 0):
    '''
    Get PS(z) from p21c LC, z range is set automatically ensuring each chunk is a cube
    -- inputs --
    H5File : H5 file for the LC
    k : k in Mpc^{-1}
    nk : number of k bins
    nz : number of z bins
    field : lightcone quantities
        0 - Tb
        1 - Tk
        2 - xH
        3 - xe
        4 - Tr
        5 - Boost
        6 - Density
    '''
    lc = p21c.LightCone.read(H5File)
    z_cube_ax = Get_p21c_lc_cube_redshifts(lc)
    zmin, zmax = np.min(z_cube_ax), np.max(z_cube_ax)
    idx1 = PL.Find_Index(x=zmin, x_axis=z_cube_ax)
    idx2 = PL.Find_Index(x=zmax, x_axis=z_cube_ax) + 1
    dn = int(np.floor((idx2 - idx1)/nz))
    z_ax = z_cube_ax[idx1:idx2:dn]
    nz_ = len(z_ax)
    for idx in tqdm.tqdm(range(len(z_ax)), desc = 'Computing lightcone PS', disable = not show_status):
        z = z_ax[idx]
        r = Get_p21c_PSk_Kernel(z=z,field=field,nk=nk,lc=lc)
        # k should be all the same
        if idx == 0:
            k_ax = r['k']
            nk_ = len(k_ax)
            PS_ax = np.zeros([nz_, nk_])
        k_ax_ = r['k']
        if np.sum(np.abs(k_ax - k_ax_)) > 1e-40:
            print('Something is wrong with k')
        PS_ax[idx, :] = r['ps'][:]
    # 2D finished, saving data
    if not output_file==None:
        np.savez(output_file, z=z_ax, k=k_ax, ps=PS_ax)
    
    # Now interpolate for 1D to get Ps(z)

    idx1 = PL.Find_Index(k, k_ax)
    idx2 = idx1+1
    x = np.log10(k)
    x1 = np.log10(k_ax[idx1])
    x2 = np.log10(k_ax[idx2])
    y1 = np.log10(PS_ax[:, idx1])
    y2 = np.log10(PS_ax[:, idx2])
    # we are interpolating in log and we don't 0 to become -inf
    y1[y1 < -150] = -150
    y2[y2 < -150] = -150
    LPS = (y2 - y1)*(x - x1)/(x2 - x1) + y1
    psz = 10**LPS
    result = {'z' : z_ax, 'psz': psz,
              'ps_2D' : PS_ax, 'k_ax' : k_ax}
    return result

def Read_P21c_PS2D(FixZ=0, z=10, k=0.1, npz_file='/tmp/tmp.npz'):
    '''
    Read ps as a function of z or k for 2D ps generated by Get_P21c_PS or Get_P21c_Coeval_cache_PS
    '''
    d = np.load(npz_file)
    z_ax = d['z']
    k_ax = d['k']
    PS_ax = d['ps']
    
    if FixZ:
        idx1 = PL.Find_Index(z,z_ax)
        idx2 = idx1 + 1
        z1 = z_ax[idx1]
        z2 = z_ax[idx2]
        y1 = np.log10(PS_ax[idx1,:])
        y2 = np.log10(PS_ax[idx2,:])
        y1[y1<-150] = -150
        y2[y2<-150] = -150
        y = (y2-y1)*(z-z1)/(z2-z1)+y1
        ps = 10**y
        return k_ax, ps
    else:
        lk_ax = np.log10(k_ax)
        x = np.log10(k)
        idx1 = PL.Find_Index(k,k_ax)
        idx2 = idx1 + 1
        x1 = lk_ax[idx1]
        x2 = lk_ax[idx2]
        y1 = np.log10(PS_ax[:, idx1])
        y2 = np.log10(PS_ax[:, idx2])
        y1[y1<-150] = -150
        y2[y2<-150] = -150
        y = (y2-y1)*(x-x1)/(x2-x1)+y1
        ps = 10**y
        return z_ax, ps

def PS2D_2_PS1D(
        kpe,
        kpa,
        PS, # [idx_kpe, idx_kpa]
        ReFine = 1,
        OverWriteK = 0,
        newk = np.logspace(-1, 1, 20),
        nk = 10,
        # PS_format = None,
        binning = 'log'):
    '''
    Convert 2D cylindrical PS to 1D PS
    Auto binning : 
        if kpe == kpa then use diagonal as newk
        otherwise use min and max
    ---- inputs ----
    kpe : perpendicular k
    kpa : parallel k
    PS : 2D array for dimensionless PS, index = [idx_kpe, idx_kpa]
    ReFine : do you wannt refine the bins?
        0 - no refining, give raw k and ps
        1 - use refining, e.g. ps modes for k in [0.5, 1.5] are assigned to k=1
    OverWriteK : use k provided in newk for binning
    newk : costume k for binning
    nk : if not OverWriteK (auto-set k), this is number of k bins
    PS_format : string, PS 2D index
        t21ct - PS[idx_kpe, idx_kpa], like tools_21cm / py21cmfast_tools outputs
        SKA_DC - PS[idx_kpa, idx_kpe], like SKA DC dataset
    binning : 
        log - use log k bins, e.g. k = [0.1, 1, 10]
        linear - use linear k bins, e.g. k = [0.1, 0.2, 0.3]
    '''
    # if len(kpe) == len(kpa) and PS_format == None: raise Exception('kpe and kpa has same size, I wont be able to auto-detect PS dimension')
    def Find_Index(x, xax):
        # Find index of x in xax (linear distributed), if not in +-0.5dx then give nan
        dx_left = xax[1] - xax[0]
        dx_right = xax[-1] - xax[-2]
        if x < xax[0] - dx_left/2 or x > xax[-1] + dx_right/2:
            return np.nan
        dist = (x - xax)**2
        idx = np.argmin(dist)
        return idx
    k = []
    psk = []
    for ide, kpe_ in enumerate(kpe):
        for ida, kpa_ in enumerate(kpa):
            k_ = (kpe_**2 + kpa_**2)**0.5
            PS_ = PS[ide, ida]
            psk.append(PS_)
            k.append(k_)
    k, psk = np.array(k), np.array(psk)
    
    # Remove_nan:
    k1 = []
    psk1 = []
    for idx, k_ in enumerate(k):
        if not np.isnan(k_):
            k1.append(k_)
            psk1.append(psk[idx])
    k1, psk1 = np.array(k1), np.array(psk1)
    if not ReFine: # If you don't wanna get binning, return k&ps now
        return k1, psk1
    
    # Get automatic k bins, ignored if OverWriteK
    lk1 = np.log10(k1)
    if binning == 'log':
        auto_bin = np.linspace(np.min(lk1), np.max(lk1), nk)
    elif binning == 'linear':
        auto_bin = np.linspace(np.min(k1), np.max(k1), nk)
    
    if binning == 'log':
        lk2 = np.log10(newk) if OverWriteK else auto_bin
        LenK = len(lk2)
        psk2 = np.zeros(LenK)
        count = np.zeros(LenK)
        for idx, lk_ in enumerate(lk1):
            idx_min = Find_Index(x = lk_, xax = lk2)
            if not np.isnan(idx_min):
                psk2[idx_min] += psk1[idx]
                count[idx_min] += 1
        psk2 = psk2/count
        k2 = 10**lk2
    elif binning == 'linear':
        k2 = newk if OverWriteK else auto_bin
        LenK = len(k2)
        psk2 = np.zeros(LenK)
        count = np.zeros(LenK)
        for idx, k_ in enumerate(k1):
            idx_min = Find_Index(x = k_, xax = k2)
            if not np.isnan(idx_min):
                psk2[idx_min] += psk1[idx]
                count[idx_min] += 1
        psk2 = psk2/count
        if True in np.isnan(psk2): raise Exception("ps is nan, count = {:.0f}".format(count[idx_min]))
    
    return k2, psk2

def Compute_nd_PS_t21c(
        box,
        box_length = [np.nan, np.nan, np.nan],
        do_1D = True,
        do_2D = True,
        kbins_1D = np.logspace(-1.7, 0.5, 40),
        kbins_2D = [np.linspace(0.02, 0.6, 10), np.linspace(0.02, 0.6, 10)],
        binning = 'log'):
    '''
    An EZ interface to compute 1D & 2D PS using tools21cm. The original is mature enough already, I am putting this here just so I know I have this
    ---- inputs ----
    box : 3D box array for which you want the PS
    box_length : length of box, list or array of form [BOX_LEN_1, BOX_LEN_2, BOX_LEN_3]
    do_1D : compute 1D spherical PS
    do_2D : compute 2D cylindrical PS
    kbins_1D : array, k bin centers for 1D PS
    kbins_2D : list for [k_perp, k_par], both k_perp and k_par are arrays
    binning : string, binning for k
        log
        linear
    nk_1d : 1D PS k length, not related to kbins_1D
    '''
    out = {}
    # Get k1d bin edges following Adele's advice, k must be linearly spaced
    def Find_k_bin_edges(k_axis):
        nk = len(k_axis)
        dks = np.abs(k_axis[0:nk-1] - k_axis[1:nk])
        dk_mean = np.mean(dks)
        dk1 = dks[1]
        if np.abs(1-dk1/dk_mean) > 1E-2:
            print('Something is wrong with k2d axis')
            print(dk1, dk_mean)
            raise Exception('kbins_1D should be linearly spaced, dev version for SKA DC')
        k_edges = np.linspace(np.min(k_axis) - dk_mean/2, np.max(k_axis) + dk_mean/2, nk + 1)
        return k_edges
    
    if do_2D:
        # print('kbins_2D[0]:', kbins_2D[0])
        kpe_edges = Find_k_bin_edges(kbins_2D[0])
        kpa_edges = Find_k_bin_edges(kbins_2D[1])
        pk2d, kpe, kpa = t21c.power_spectrum.power_spectrum_2d(
            input_array = box, 
            kbins = [kpe_edges, kpa_edges], 
            binning = binning,
            box_dims = [box_length[0], box_length[1], box_length[2]],
            return_modes = False, 
            nu_axis = 2, 
            window=None)
        # Check that returned kpe and kpa are the same as input
        if np.mean(np.abs(1 - kpe/kbins_2D[0])) > 1E-2 or np.mean(np.abs(1 - kpa/kbins_2D[1])) > 1E-2:
            raise Exception('Unexpected kpe and kpa bins')
        # ps2d is not dimensionless and it has index [kpe, kpa]
        ps2d = 0*pk2d
        for ide, kpe_ in enumerate(kpe):
            for ida, kpa_ in enumerate(kpa):
                k_ = (kpe_**2 + kpa_**2)**0.5
                ps2d[ide, ida] = pk2d[ide, ida] * k_**3 / (2 * np.pi**2)
        out['pk2d'] = pk2d
        out['ps2d'] = ps2d
        out['kpe'] = kpe
        out['kpa'] = kpa
        
        # While we are at it, get 1D PS as well
        # Currently this module is used for SKA DC which has linear binning
        k, ps1d = PS2D_2_PS1D(
            kpe = kpe,
            kpa = kpa,
            PS = ps2d,
            ReFine = 1,
            OverWriteK = 1,
            newk = kbins_1D,
            binning = binning)
        out['ps1d_derived'] = ps1d
        out['k1d_derived'] = k
    if do_1D:
        k1d_bin_edges = Find_k_bin_edges(kbins_1D)
        PS, ks = t21c.power_spectrum.power_spectrum_1d(
            input_array_nd = box,
            kbins = k1d_bin_edges,
            box_dims = [box_length[0], box_length[1], box_length[2]],
            binning = 'linear')
        PS = PS*ks**3/(2 * np.pi**2)
        #if binning == 'log': PS = np.interp(x = np.log(kbins_1D), xp = np.log(ks), fp = PS, left = np.nan, right = np.nan)
        #if binning == 'linear': PS = np.interp(x = kbins_1D, xp = ks, fp = PS, left = np.nan, right = np.nan)
        # At this point we should get kbins_1D as our k
        if np.mean(np.abs(1 - kbins_1D/ks)) > 1E-2: raise Exception('Unexpected k')
        out['ps1d'] = PS
        out['k1d'] = kbins_1D
    return out

def Read_p21c_cache(path, z, field = 'Tb', result_type = 'box'):
    '''
    Read 21cmFAST cache files
    ---- inputs ----
    result_type: what to return
        redshift - redshifts for coeval boxes
        box - coeval box at z
        file - cache file at z
    '''
    # get_z
    if field == 'Tb':
        head, name = 'BrightnessTemp_', 'BrightnessTemp/brightness_temp'
    elif field == 'nion':
        head, name = 'HaloBox_', 'HaloBox/n_ion'
    elif field == 'density':
        head, name = 'PerturbedField_', 'PerturbedField/density'
    elif field == 'xH':
        head, name = 'IonizedBox_', 'IonizedBox/xH_box'
    elif field == 'HaloField':
        head = 'HaloField_'
    if field == 'Tb' or result_type == 'redshift': head = 'BrightnessTemp_'
    Files = os.listdir(path)
    FL = []
    for file in Files:
        if head in file and '.h5' in file:
            if field == 'HaloField':
                # We don't want PerturbHaloField_*.h5
                if file[0] == 'H':
                    FL.append(path + file)
            else:
                FL.append(path + file)
    zax = []
    for file in FL:
        H5F = h5py.File(file, 'r')
        zax.append(H5F.attrs['redshift'])
        H5F.close()
    
    # Ok now z and FL are matching pairs
    zax = np.array(zax)
    if result_type == 'redshift': 
        return np.sort(zax)[::-1]
    else:
        dif = np.abs(z - zax)
        idx = np.argmin(dif)
        if dif[idx] > 1E-2: raise Exception('Cannot find requested file from cache')
        # Reading data
        file = FL[idx]
        if result_type == 'file': 
            return file
        else:
            H5F = h5py.File(file, 'r')
            box = H5F[name][:]
            H5F.close()
            return box
