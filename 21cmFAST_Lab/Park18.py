
FileName = '/Users/cangtao/Desktop/21cmFAST-data/Park18.h5'
redshift = 4.9
max_redshift = 35
LC_Quantities = ('brightness_temp','Ts_box','xH_box','Tk_box')
GLB_Quantities = ('brightness_temp','Ts_box','xH_box','dNrec_box','z_re_box', 'Gamma12_box','J_21_LW_box','density','Tk_box','Fcoll')

import py21cmfast as p21c
import time, os

user_params = p21c.UserParams(
  HII_DIM = 200,
  N_THREADS = 1,
  # USE_RELATIVE_VELOCITIES = False,
  USE_INTERPOLATION_TABLES = True,
  # FAST_FCOLL_TABLES = False,
  # DIM = None,
  BOX_LEN = 300,
  # USE_FFTW_WISDOM = False,
  # HMF = 1,
  # POWER_SPECTRUM = 0,
  # PERTURB_ON_HIGH_RES = False,
  # NO_RNG = False,
  # USE_2LPT = True,
  # MINIMIZE_MEMORY = False,
  )

astro_params = p21c.AstroParams(
  F_STAR10 = -1.301,
  L_X = 40.5,
  NU_X_THRESH = 500.0,
  t_STAR = 0.5,
  ALPHA_STAR = 0.5,
  M_TURN = 8.7,
  # HII_EFF_FACTOR = 30.0,
  # R_BUBBLE_MAX = None,
  # ION_Tvir_MIN = 4.69897,
  # F_ESC10 = -1.0,
  # F_ESC7_MINI = -2.0,
  # ALPHA_ESC = -0.5,
  # X_RAY_SPEC_INDEX = 1.0,
  # X_RAY_Tvir_MIN = None,
  # F_H2_SHIELD = 0.0,
  # N_RSD_STEPS = 20,
  )

flag_options = p21c.FlagOptions(
  USE_MASS_DEPENDENT_ZETA = True,
  INHOMO_RECO = True,
  USE_TS_FLUCT = True,
  # SUBCELL_RSD = False,
  # M_MIN_in_Mass = False,
  # PHOTON_CONS = False,
  # FIX_VCB_AVG = False,
)

# ---- Initialise ----
start_time = time.time()
lc = p21c.run_lightcone(
  redshift=redshift, 
  max_redshift=max_redshift,
  astro_params=astro_params, 
  flag_options=flag_options,
  user_params = user_params,
  lightcone_quantities=LC_Quantities,
  global_quantities=GLB_Quantities
  )
end_time = time.time()
print('--------RunTime--------')
print(end_time - start_time)

try:
  os.remove(FileName)
except:
  pass
lc.save(FileName)
