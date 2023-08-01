
import numpy as np
from powerbox.tools import get_power
import py21cmfast as p21c
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

def powerspectra(
        FileName = '/home/dm/gaolq/cjs/21cmEZ/Park18.h5',
        n_psbins=50,
        nchunks=10,
        min_k=0.1,
        max_k=1.0,
        logk=True):
    LightCone = p21c.LightCone.read(FileName)
    data = []
    chunk_indices = list(range(0,LightCone.n_slices,round(LightCone.n_slices / nchunks),))
    BOX_LEN=LightCone.user_params.BOX_LEN
    if len(chunk_indices) > nchunks:
        chunk_indices = chunk_indices[:-1]
    chunk_indices.append(LightCone.n_slices)

    for i in range(nchunks):
        start = chunk_indices[i]
        end = chunk_indices[i + 1]
        chunklen = (end - start) * LightCone.cell_size

        power, k = compute_power(
            LightCone.brightness_temp[:, :, start:end],
            (BOX_LEN, BOX_LEN, chunklen),
            n_psbins,
            log_bins=logk,
        )
        data.append({"k": k, "delta": power * k ** 3 / (2 * np.pi ** 2)})
    return data

t1 = time.time()
a = powerspectra()
t2 = time.time()
print('time = ', t2 - t1)

