'''
Some useful python functions fro 21cmFAST related calculations, to be able to use this from any locations, add this to your bash_profile:
export PYTHONPATH=Current_Path:${PYTHONPATH}
Functions:
1 - compute_power
2 - PowerSpectra
3 - PowerSpectra_Coeval
4 - LightCone_Postprocessing
5 - Validate_Inputs
6 - HMF
7 - Hubble
8 - MUV2Mh
9 - SFRD
10 - UVLF
11 - Radio_Temp_Astro
'''

import numpy as np
import h5py, os
from powerbox.tools import get_power
import py21cmfast as p21c
from PyLab import *
try:
  from hmf import MassFunction
except:
   pass
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import time

# ---- Power spectra ----
def compute_power(
   box,
   length,
   SizeK,
   log_bins=True,
   ignore_kperp_zero=True,
   ignore_kpar_zero=False,
   ignore_k_zero=False,
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
        bins=SizeK,
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

# ---- 2 ----
def PowerSpectra(FileName = '/home/dm/gaolq/cjs/21cmEZ/Park18.h5', 
                SizeK=50,
                SizeZ=30,
                k_vec = np.linspace(0.2, 1, 20),
                z_vec = np.linspace(14, 28, 40),
                max_k=2,
                logk=True,
                Flatten = False):
    '''
    Calculate Power Spectra for LightCone object
    This is the beta version, check again
    ---- Inputs ----
    LightCone: p21c LC object, can come from p21c.run_lightcone or existing h5
    SizeK: Number of k values you want
    SizeZ: Number of z values you want
    DataFile: File name for Power spectra
    '''

    LightCone = p21c.LightCone.read(FileName)
    lightcone_redshifts=LightCone.lightcone_redshifts
    MaxNz=LightCone.n_slices # Number of slices along z axis
    BOX_LEN=LightCone.user_params.BOX_LEN
    HII_DIM=LightCone.user_params.HII_DIM
    CellSize=BOX_LEN/HII_DIM
    min_k=1/BOX_LEN
    #max_k=1/CellSize

    # z indexes for requested redshifts
    Interval=max(round(MaxNz / SizeZ),1)
    IdxZs = list(range(0,MaxNz,Interval,))
    
    if len(IdxZs) > SizeZ:# This might happen due to rounding
        IdxZs = IdxZs[:-1]
    if IdxZs[-1] != MaxNz:# Compute the last slice too
        IdxZs.append(MaxNz)

    NZ=len(IdxZs) # Can be different from SizeZ now
    Z_Axis=np.zeros((NZ-1),dtype=float)

    for IdxZ in range(NZ-1):# Must -1 or Idx2 will get index error
      Z_Axis[IdxZ]=lightcone_redshifts[IdxZs[IdxZ]]
    
    Ps_vec = np.empty((len(z_vec), len(k_vec)))
    Ps_Tab = np.empty((NZ - 1, len(k_vec)))

    for IdxZ in range(NZ-1):# Must -1 or Idx2 will get index error
        Idx1 = IdxZs[IdxZ]
        Idx2 = IdxZs[IdxZ + 1]
        chunklen = (Idx2 - Idx1) * LightCone.cell_size

        Pk, k = compute_power(
            LightCone.brightness_temp[:, :, Idx1:Idx2],
            (BOX_LEN, BOX_LEN, chunklen),
            SizeK,
            log_bins=logk
        )
        # remove NaN section
        NanIdx = np.isnan(Pk)
        nk = len(NanIdx)
        
        NanLen = 0
        for kid in np.arange(0, nk):
           if NanIdx[kid]:
              NanLen = kid + 1
        if NanLen > 0:
          NanID = np.arange(0, NanLen)
          k = np.delete(k, NanID)
          Pk = np.delete(Pk, NanID)
          # print('k length = ', len(k))

        Ps = Pk * k ** 3 / (2 * np.pi ** 2)
        Psk = np.interp(k_vec, k, Ps)
        Ps_Tab[IdxZ, :] = Psk[:]
    
    # Now interpolate z axis
    pz_axis = np.linspace(0, len(Z_Axis))
    pz_vec = np.linspace(0, len(z_vec))

    for kid in np.arange(0, len(k_vec)):
      pz_axis = Ps_Tab[:,kid]
      pz_vec = np.interp(z_vec, Z_Axis, pz_axis, left = np.nan, right = np.nan)
      nan_id = np.isnan(pz_vec)
      if True in nan_id:
        print('Z_Axis range:', np.min(Z_Axis), np.max(Z_Axis))
        raise Exception('NaN found!')
      Ps_vec[:, kid] = pz_vec[:]
    
    if not Flatten:
       return Ps_vec

    # Now Flatten it
    r = np.linspace(0, 1, len(k_vec)*len(z_vec))
    idx = 0
    for kid in np.arange(0, len(k_vec)):
       for zid in np.arange(0, len(z_vec)):
          r[idx] = Ps_vec[zid, kid]
          idx = idx + 1

    '''
    # All data now acquired, saving to file
    h5f=h5py.File(DataFile,'a')
    h5f.create_dataset('PowerSpectra', data=powerspectra)
    h5f.create_dataset('PowerSpectra_Redshifts', data=Z_Axis)
    h5f.close()
    '''

    return r

def Find_Ps_idx(kid = 3,
                zid = 2,
                k_vec = np.linspace(0.2, 1, 20),
                z_vec = np.linspace(14, 28, 40)
                ):
   nk = len(k_vec)
   nz = len(z_vec)
   r = kid*nz + zid
   return r

def Find_Psz(Ps_Tab,
             k = 0.5,
             k_vec = np.linspace(0.2, 1, 20),
             z_vec = np.linspace(14, 28, 40),
             ):
    nk = len(k_vec)
    nz = len(z_vec)
    kid = Find_Index(x = k, x_axis = k_vec)
    r = np.linspace(0, 1, nz)
    for zid in np.arange(0, nz):
      idx = Find_Ps_idx(kid = kid, zid = zid, k_vec = k_vec, z_vec = z_vec)
      r[zid] = Ps_Tab[idx]
    return r

def Find_Psk(Ps_Tab,
             z = 17.5,
             k_vec = np.linspace(0.2, 1, 20),
             z_vec = np.linspace(14, 28, 40),
             ):
    nk = len(k_vec)
    nz = len(z_vec)
    zid = Find_Index(x = z, x_axis = z_vec)
    r = np.linspace(0, 1, nk)
    for kid in np.arange(0, nk):
      idx = Find_Ps_idx(kid = kid, zid = zid, k_vec = k_vec, z_vec = z_vec)
      r[zid] = Ps_Tab[idx]
    return r


def PowerSpectra_Coeval(Coeval, Field=1, SizeK=50, max_k=2,logk=True, DataFile='/home/jcang/21cmFAST-data/CoevalPower.h5'):
    '''
    Calculate Power Spectra for LightCone object
    ---- Inputs ----
    Coeval: p21c Coeval object, can come from p21c.run_coeval or existing h5
    Field: Choose field, default is brightness_temp, can also be xH or density, etc
    SizeK: Number of k values you want
    DataFile: File name for Power spectra
    '''
    BOX_LEN=Coeval.user_params.BOX_LEN
    HII_DIM=Coeval.user_params.HII_DIM
    CellSize=BOX_LEN/HII_DIM
    min_k=1/BOX_LEN
    if Field==1:
        DataBox=Coeval.brightness_temp

    Pk, k = compute_power(
        DataBox,
        (BOX_LEN,BOX_LEN,BOX_LEN),
        SizeK,
        log_bins=logk
        )
    Ps=Pk * k ** 3 / (2 * np.pi ** 2)
    h5f=h5py.File(DataFile,'a')
    h5f.create_dataset('PowerSpectra/k', data=k)
    h5f.create_dataset('PowerSpectra/Ps', data=Ps)
    h5f.close()

def LightCone_Postprocessing(LightCone, FileName):
    # 1 --save power spectra
    PowerSpectra(LightCone,DataFile=FileName)
    # 2 -- save lightcone_redshifts
    lightcone_redshifts=LightCone.lightcone_redshifts
    h5f=h5py.File(FileName, 'a')
    h5f.create_dataset('lightcone_redshifts', data=lightcone_redshifts)
    h5f.close()

def Validate_Inputs(user_params, flag_options):
    
    if flag_options.USE_RADIO_MCG:
        if not flag_options.USE_MINI_HALOS:
            raise Exception('USE_RADIO_MCG requires USE_MINI_HALOS')
        if not user_params.USE_INTERPOLATION_TABLES:
            raise Exception('USE_RADIO_MCG requires USE_INTERPOLATION_TABLES')
        if not user_params.FAST_FCOLL_TABLES:
            raise Exception('USE_RADIO_MCG requires FAST_FCOLL_TABLES')
        if not user_params.USE_RELATIVE_VELOCITIES:
            raise Exception('USE_RADIO_MCG requires USE_RELATIVE_VELOCITIES')
        if not flag_options.USE_MASS_DEPENDENT_ZETA:
            raise Exception('USE_RADIO_MCG requires USE_MASS_DEPENDENT_ZETA')
        if not flag_options.INHOMO_RECO:
            raise Exception('USE_RADIO_MCG requires INHOMO_RECO')
        if not flag_options.USE_TS_FLUCT:
            raise Exception('USE_RADIO_MCG requires USE_TS_FLUCT')
    
    if flag_options.USE_RADIO_ACG:
        if not flag_options.INHOMO_RECO:
            raise Exception('USE_RADIO_ACG requires INHOMO_RECO')
        if not flag_options.USE_TS_FLUCT:
            raise Exception('USE_RADIO_ACG requires USE_TS_FLUCT')


def HMF(z=0,
        model = 1,
        Mmin = 1e2,
        Mmax = 1E18,
        nm = 100,
        POWER_SPECTRUM = 0,
        Print_Result = False):
  '''An interface with hmf package
  -- inputs --
     z : z
     model : hmf model, follows 21cmFAST convention
             0 - PS
             1 - ST
     Mmin : minimum halo mass in Msun
     Mmax : maximum halo mass in Msun
     nm : number of mass points
     POWER_SPECTRUM : Transfer Function
  -- outputs --
  m : mh in Msun
  dndm : dn/dm in Mpc^-3 Msun^-1
  '''
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
  File = "/Users/cangtao/cloud/Matlab/21cmFAST/SanityCheck/HMF/data/HMF_hmf.txt"
  if Print_Result:
    try:
      os.remove(File)
    except:
      pass
    F=open(File,'w')
    print("m    dndm", file = F)
    for id in np.arange(0,len(m0)):
      print("{0:.8E}".format(m0[id]), "    {0:.8E}".format(dndm0[id]), file=F)
    F.close()
  # Ensure that size of m matches the input nm
  if len(m0) != nm:
    m = np.logspace(np.log10(Mmin), np.log10(Mmax), nm)
    dndm = spline(m0, dndm0)(m)
    dndm[0] = dndm0[0]
    dndm[-1] = dndm0[-1]
  else:
    m, dndm = m0, dndm0
  return m,dndm

def Hubble(z=0):
  OmM = 0.3111
  OmL = 0.6888
  h = 0.6766
  H0 = h*3.240755744239557E-18
  zp3 = pow(1+z, 3)
  r = H0 * np.sqrt(OmL + OmM * zp3)
  return r
  
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
  H = Hubble(z)
  m1 = t_STAR * OmM * pow(10, 10 * ALPHA_STAR) * SFR / (H * f10 * OmB)
  r = pow(m1, 1/(ALPHA_STAR+1))
  return r

def SFRD(
    zmin = 5.0,
    zmax = 30.0,
    nz = 30,
    t_STAR = 0.5,
    ALPHA_STAR = 0.5,
    Lf10 = -1.3010,
    LMturn = 8.699,
    hmf_model = 1,
    Transfer_model = 0,
    nm = 10000):
  ''' Get SFRD in Msun yr^-1 Mpc^-3,
  default settings reproduces Park 18 results, see 1809.08995'''

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
  year = 31557600
  # Starting
  z_array = np.linspace(zmin,zmax,nz)
  r = np.linspace(zmin,zmax,nz)
  f_duty = np.linspace(1,100,nm)
  SFR = np.linspace(1,100,nm)
  for zid in np.arange(0,nz):
    z = z_array[zid]
    H = Hubble(z)
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
  return z_array, r

def UVLF(
    M1 = -22,
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
  H = Hubble(z)
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
    zmin = 5,
    zmax = 30,
    nz = 100,
    zcut = 17,
    z_axis = np.linspace(5, 30, 100),
    SFRD_axis = np.logspace(-1, -8.64, 100),
    Use_SFRD_Table = False
    ):
  '''
  Get Radio Temp for astrophysical sources
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

  z_new = np.linspace(z_axis[0], z_axis[-1], 10000)
  SFRD_new = np.interp(z_new, z_axis, SFRD_axis)  
  SFRD_SI = SFRD_new * Msun / Yr / pow(Mpc, 3)
  H = Hubble(z_new)
  z = np.linspace(zmin, zmax, nz)
  T = np.linspace(0, 1, nz)
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

'''
from PyLab import *
File = '/Users/cangtao/cloud/Library/PyLab/Curve_Data/1803.03272.fig5c.light_blue_up.txt'
File = '/Users/cangtao/cloud/Library/PyLab/Curve_Data/1803.03272.fig5c.dark_blue_up.txt'
#File = '/Users/cangtao/cloud/Library/PyLab/Curve_Data/1803.03272.fig5c.dark_blue_low.txt'
#File = '/Users/cangtao/cloud/Library/PyLab/Curve_Data/1803.03272.fig5c.light_blue_low.txt'

z_axis, SFRD_Tab = Read_Curve(
  File = File,
  Convert_y = True
  )

z, T = Radio_Temp_Astro(fR = pow(10,3.5),
                        z_axis= z_axis,
                        SFRD_axis= SFRD_Tab
                         )
T17 = np.interp(17, z, T)
print(T17)
plt.semilogy(z, T)
plt.show()

f = '/home/dm/gaolq/cjs/21cmFAST_cache/MCMC_cache_10/LC_0.002644_-1.258330_-1.220660_4.381547E+01_0.205800.h5'

t1 = time.time()
a = PowerSpectra(FileName = f)
t2 = time.time()
print(a)

print('time = ', t2 - t1)
'''
