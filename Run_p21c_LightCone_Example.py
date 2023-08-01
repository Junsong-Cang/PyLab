
FileName = '/Users/cangtao/Desktop/21cmFAST-data/test_lc_HiRes.h5'
redshift = 7.0
max_redshift = 35
LC_Quantities = ('brightness_temp','Ts_box','xH_box','Tk_box','Trad_box')
GLB_Quantities = ('brightness_temp','Ts_box','xH_box','dNrec_box','z_re_box','Gamma12_box','J_21_LW_box','density','Trad_box','Tk_box','Fcoll')

from p21c_tools import *
import py21cmfast as p21c
import time

user_params = p21c.UserParams(
  HII_DIM = 60,
  N_THREADS = 1,
  USE_RELATIVE_VELOCITIES = False,
  USE_INTERPOLATION_TABLES = True,
  FAST_FCOLL_TABLES = False,
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
  fR = 5.0,
  fR_mini = -10.0,
  F_STAR10 = -1.25,
  F_STAR7_MINI = -2.0,
  L_X = 40.0,
  L_X_MINI = 40.0,
  Radio_Zmin = 20,
  NU_X_THRESH = 500.0,
  t_STAR = 0.5,
  aR = 0.62,
  ALPHA_STAR = 0.5,
  ALPHA_STAR_MINI = 0.5,
  M_TURN = 8.7,
  aR_mini = 0.62,
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
  # F_ESC10 = -1.0,
  # F_ESC7_MINI = -2.0,
  # ALPHA_ESC = -0.5,
  # X_RAY_SPEC_INDEX = 1.0,
  # X_RAY_Tvir_MIN = None,
  # F_H2_SHIELD = 0.0,
  # N_RSD_STEPS = 20,
  )

flag_options = p21c.FlagOptions(
  USE_RADIO_ACG = True,
  USE_RADIO_MCG = False,
  USE_MINI_HALOS = False,
  USE_MASS_DEPENDENT_ZETA = True,
  INHOMO_RECO = True,
  USE_TS_FLUCT = True,
  USE_RADIO_PBH = False,
  USE_HAWKING_RADIATION = False,
  # USE_HALO_FIELD = False,
  # SUBCELL_RSD = False,
  # M_MIN_in_Mass = False,
  # PHOTON_CONS = False,
  # FIX_VCB_AVG = False,
)

# ---- Initialise ----
Validate_Inputs(user_params, flag_options)
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
  lc.save(FileName)
except:
  # Remove pre-existing file
  os.remove(FileName)
  lc.save(FileName)

# LightCone_Postprocessing(lc, FileName)
