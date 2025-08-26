FileName = '/Users/cangtao/Desktop/21cmFAST-data/EOS_2021_no_mini.h5'
redshift = 3.0
max_redshift = 35
LC_Quantities = ('brightness_temp','Ts_box','xH_box','Tk_box','Trad_box')
GLB_Quantities = ('brightness_temp','Ts_box','xH_box','dNrec_box','z_re_box','Gamma12_box','J_21_LW_box','density','Trad_box','Tk_box','Fcoll')

import py21cmfast as p21c
import time, os

user_params = p21c.UserParams(
  HII_DIM = 50,
  N_THREADS = 1,
  USE_RELATIVE_VELOCITIES = False,
  USE_INTERPOLATION_TABLES = True,
  FAST_FCOLL_TABLES = True,
  # DIM = None,
  # BOX_LEN = 500,
  # USE_FFTW_WISDOM = False,
  # HMF = 1,
  # POWER_SPECTRUM = 0,
  # PERTURB_ON_HIGH_RES = False,
  # NO_RNG = False,
  # USE_2LPT = True,
  # MINIMIZE_MEMORY = False,
  )

astro_params = p21c.AstroParams(
  # fR = -10.0,
  # fR_mini = -10.0,
  F_STAR10 = -1.25,
  F_STAR7_MINI = -2.5,
  L_X = 40.5,
  L_X_MINI = 40.5,
  # Radio_Zmin = 0,
  NU_X_THRESH = 500.0,
  t_STAR = 0.5,
  # aR = 0.62,
  ALPHA_STAR = 0.5,
  ALPHA_STAR_MINI = 0,
  # M_TURN = 8.7,
  # aR_mini = 0.62,
  A_LW = 2.0,
  BETA_LW = 0.6,
  A_VCB = 1.0,
  BETA_VCB = 1.8,
  # mbh = 1,
  # fbh = -120,
  # bh_aR = 0.6,
  # bh_fX = 0.1,
  # bh_fR = 1,
  # bh_lambda = 0.1,
  # bh_Eta = 0.1,
  # bh_spin = 0.0,
  # HII_EFF_FACTOR = 30.0,
  # R_BUBBLE_MAX = None,
  # ION_Tvir_MIN = 4.69897,
  F_ESC10 = -1.35,
  F_ESC7_MINI = -1.35,
  ALPHA_ESC = -0.3,
  # X_RAY_SPEC_INDEX = 1.0,
  # X_RAY_Tvir_MIN = None,
  # F_H2_SHIELD = 0.0,
  # N_RSD_STEPS = 20,
  )

flag_options = p21c.FlagOptions(
  # USE_RADIO_ACG = False,
  # USE_RADIO_MCG = False,
  USE_MINI_HALOS = False,
  USE_MASS_DEPENDENT_ZETA = True,
  INHOMO_RECO = True,
  USE_TS_FLUCT = True,
  # USE_RADIO_PBH = False,
  # USE_HAWKING_RADIATION = False,
  # USE_HALO_FIELD = False,
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
