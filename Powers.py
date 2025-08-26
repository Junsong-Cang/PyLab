"""Code to calculate the 1D and 2D power spectrum of a lightcone."""

import numpy as np
from powerbox.tools import (
    _magnitude_grid,
    above_mu_min_angular_generator,
    get_power,
    ignore_zero_ki,
    power2delta,
    regular_angular_generator,
)

def Compute_nd_PS_p21ct(
    box,
    box_length = [np.nan, np.nan, np.nan],
    calc_2d = True,
    kpe_bins = 50,
    kpar_bins = None,
    log_bins = True,
    k_weights = ignore_zero_ki,
    postprocess = False,
    crop = None,
    calc_1d = False,
    nbins_1d = 14,
    mu = None,
    bin_ave = True,
    interp = None,
    prefactor_fnc = power2delta,
    interp_points_generator = None,
):
    '''
    Calculate power spectra for a box, codes are adapted from py21cmfast_tools to allow for more flexible chunking
    ---- Inputs ----
    box
        A 3D array for which we wante to calculate PS
    box_length
        Side length of the box in cMpc, an array of form [BOX_LEN_1, BOX_LEN_2, BOX_LEN_3] for each dimension of box
    calc_2d
        Bool, if True, calculate the 2D power spectrum.
    kpe_bins
        Int or array, optional.
        If int, gives the number of bins to use for the kperp axis of the 2D PS.
        If array, sets the exact kperp axis of the 2D PS.
    k_weights : callable, optional
        A function that takes a frequency tuple and returns
        a boolean mask for the k values to ignore.
        See powerbox.tools.ignore_zero_ki for an example
        and powerbox.tools.get_power documentation for more details.
        Default is powerbox.tools.ignore_zero_ki, which excludes
        the power any k_i = 0 mode.
        Typically, only the central zero mode |k| = 0 is excluded,
        in which case use powerbox.tools.ignore_zero_absk.
    postprocess : bool, optional
        If True, postprocess the 2D PS.
        This step involves cropping out empty bins and/or log binning the kpar axis.
    kpar_bins : int or np.ndarray, optional
        Affects only the postprocessing step.
        The number of bins or the bin edges to use for binning the kpar axis.
        If None, produces 16 bins.
    log_bins : bool, optional
        Affects only the postprocessing step. If True, log bin the kpar axis.
    crop : list, optional
        Affects only the postprocessing step.
        The crop range for the (log-binned) PS. If None, crops out only the empty bins.
    calc_1d : bool, optional
        If True, calculate the 1D power spectrum.
    nbins_1d : int, optional
        The number of bins on which to calculate 1D PS.
    calc_global : bool, optional
        If True, calculate the global brightness temperature.
    mu : float, optional
        The minimum value of
        :math:`\\cos(\theta), \theta = \arctan (k_\\perp/k_\\parallel)`
        for all calculated PS.
        If None, all modes are included.
    bin_ave : bool, optional
        If True, return the center value of each kperp and kpar bin
        i.e. len(kperp) = ps_2d.shape[0].
        If False, return the left edge of each bin
        i.e. len(kperp) = ps_2d.shape[0] + 1.
    interp : str, optional
        If True, use linear interpolation to calculate the PS
        at the points specified by interp_points_generator.
        Note that this significantly slows down the calculation.
    prefactor_fnc : callable, optional
        A function that takes a frequency tuple and returns the prefactor
        to multiply the PS with.
        Default is powerbox.tools.power2delta, which converts the power
        P [mK^2 Mpc^{-3}] to the dimensionless power :math:`\\delta^2` [mK^2].
    interp_points_generator : callable, optional
        A function that generates the points at which to interpolate the PS.
        See powerbox.tools.get_power documentation for more details.
    '''
    if calc_1d:
        lc_ps_1d = []
    out = {}
    if interp:
        interp = "linear"
    
    if calc_2d:
        ps_2d, kperp, nmodes, kpar = get_power(
            box,
            (box_length[0], box_length[1], box_length[2]),
            res_ndim=2,
            bin_ave=bin_ave,
            bins=kpe_bins,
            log_bins=log_bins,
            nthreads=1,
            k_weights=k_weights,
            prefactor_fnc = prefactor_fnc,
            interpolation_method = interp,
            return_sumweights = True)
        if postprocess:
            clean_ps_2d, clean_kperp, clean_kpar, clean_nmodes = postprocess_ps(
                ps_2d,
                kperp,
                kpar,
                log_bins=log_bins,
                kpar_bins=kpar_bins,
                crop=crop.copy() if crop is not None else crop,
                kperp_modes=nmodes,
                return_modes=True,
            )
            clean_lc_ps_2d = clean_ps_2d
    if calc_1d:
        if mu is not None:
            if interp is None:
                def mask_fnc(freq, absk):
                    kz_mesh = np.zeros((len(freq[0]), len(freq[1]), len(freq[2])))
                    kz = freq[2]
                    for i in range(len(kz)):
                        kz_mesh[:, :, i] = kz[i]
                    phi = np.arccos(kz_mesh / absk)
                    mu_mesh = abs(np.cos(phi))
                    kmag = _magnitude_grid([c for i, c in enumerate(freq) if i < 2])
                    return np.logical_and(mu_mesh > mu, ignore_zero_ki(freq, kmag))
                k_weights1d = mask_fnc
            if interp is not None:
                k_weights1d = ignore_zero_ki
                interp_points_generator = above_mu_min_angular_generator(mu=mu)
        else:
            k_weights1d = ignore_zero_ki
            if interp is not None:
                interp_points_generator = regular_angular_generator()
        ps_1d, k, nmodes_1d = get_power(
            box,
            (box_length[0], box_length[1], box_length[2]),
            bin_ave=bin_ave,
            bins=nbins_1d,
            log_bins=log_bins,
            k_weights=k_weights1d,
            prefactor_fnc=prefactor_fnc,
            interpolation_method=interp,
            interp_points_generator=interp_points_generator,
            return_sumweights=True)
        lc_ps_1d.append(ps_1d)

    if calc_1d:
        out["k"] = k
        out["ps_1D"] = np.array(lc_ps_1d)
        out["Nmodes_1D"] = nmodes_1d
        out["mu"] = mu
    if calc_2d:
        out["full_kperp"] = kperp
        out["full_kpar"] = kpar
        out["full_ps_2D"] = np.array(ps_2d)
        out["full_Nmodes"] = nmodes
        if postprocess:
            out["final_ps_2D"] = np.array(clean_lc_ps_2d)
            out["final_kpar"] = clean_kpar
            out["final_kperp"] = clean_kperp
            out["final_Nmodes"] = clean_nmodes
    return out

def log_bin(ps, kperp, kpar, bins=None):
    r"""
    Log bin a 2D PS along the kpar axis and crop out empty bins in both axes.

    Parameters
    ----------
    ps : np.ndarray
        The 2D power spectrum of shape [len(kperp), len(kpar)].
    kperp : np.ndarray
        Values of kperp.
    kpar : np.ndarray
        Values of kpar.
    bins : np.ndarray or int, optional
        The number of bins or the bin edges to use for binning the kpar axis.
        If None, produces 16 bins logarithmically spaced between
        the minimum and maximum `kpar` supplied.

    """
    if bins is None:
        bins = np.logspace(np.log10(kpar[0]), np.log10(kpar[-1]), 17)
    elif isinstance(bins, int):
        bins = np.logspace(np.log10(kpar[0]), np.log10(kpar[-1]), bins + 1)
    elif isinstance(bins, (np.ndarray, list)):
        bins = np.array(bins)
    else:
        raise ValueError("Bins should be np.ndarray or int")
    modes = np.zeros(len(bins) - 1)
    new_ps = np.zeros((len(kperp), len(bins) - 1))
    for i in range(len(bins) - 1):
        m = np.logical_and(kpar >= bins[i], kpar < bins[i + 1])
        new_ps[:, i] = np.nanmean(ps[:, m], axis=1)
        modes[i] = np.sum(m)
    bin_centers = np.exp((np.log(bins[1:]) + np.log(bins[:-1])) / 2)
    return new_ps, kperp, bin_centers, modes

def postprocess_ps(
    ps,
    kperp,
    kpar,
    kpar_bins=None,
    log_bins=True,
    crop=None,
    kperp_modes=None,
    return_modes=False,
):
    """
    Postprocess a 2D PS by cropping out empty bins and log binning the kpar axis.

    Parameters
    ----------
    ps : np.ndarray
        The 2D power spectrum of shape [len(kperp), len(kpar)].
    kperp : np.ndarray
        Values of kperp.
    kpar : np.ndarray
        Values of kpar.
    kpar_bins : np.ndarray or int, optional
        The number of bins or the bin edges to use for binning the kpar axis.
        If None, produces 16 bins log spaced between the min and max `kpar` supplied.
    log_bins : bool, optional
        If True, log bin the kpar axis.
    crop : list, optional
        The crop range for the log-binned PS. If None, crops out all empty bins.
    kperp_modes : np.ndarray, optional
        The number of modes in each kperp bin.
    return_modes : bool, optional
        If True, return a grid with the number of modes in each bin.
        Requires kperp_modes to be supplied.
    """
    kpar = kpar[0]
    m = kpar > 1e-10
    if ps.shape[0] < len(kperp):
        if log_bins:
            kperp = np.exp((np.log(kperp[1:]) + np.log(kperp[:-1])) / 2.0)
        else:
            kperp = (kperp[1:] + kperp[:-1]) / 2
    kpar = kpar[m]
    ps = ps[:, m]
    mkperp = ~np.isnan(kperp)
    if kperp_modes is not None:
        kperp_modes = kperp_modes[mkperp]
    kperp = kperp[mkperp]
    ps = ps[mkperp, :]

    # Bin kpar in log
    rebinned_ps, kperp, log_kpar, kpar_weights = log_bin(
        ps, kperp, kpar, bins=kpar_bins
    )
    if crop is None:
        crop = [0, rebinned_ps.shape[0] + 1, 0, rebinned_ps.shape[1] + 1]
    # Find last bin that is NaN and cut out all bins before
    try:
        lastnan_perp = np.where(np.isnan(np.nanmean(rebinned_ps, axis=1)))[0][-1] + 1
        crop[0] = crop[0] + lastnan_perp
    except IndexError:
        pass
    try:
        lastnan_par = np.where(np.isnan(np.nanmean(rebinned_ps, axis=0)))[0][-1] + 1
        crop[2] = crop[2] + lastnan_par
    except IndexError:
        pass
    if kperp_modes is not None:
        final_kperp_modes = kperp_modes[crop[0] : crop[1]]
        kpar_grid, kperp_grid = np.meshgrid(
            kpar_weights[crop[2] : crop[3]], final_kperp_modes
        )

        nmodes = np.sqrt(kperp_grid**2 + kpar_grid**2)
        if return_modes:
            return (
                rebinned_ps[crop[0] : crop[1]][:, crop[2] : crop[3]],
                kperp[crop[0] : crop[1]],
                log_kpar[crop[2] : crop[3]],
                nmodes,
            )
    else:
        return (
            rebinned_ps[crop[0] : crop[1]][:, crop[2] : crop[3]],
            kperp[crop[0] : crop[1]],
            log_kpar[crop[2] : crop[3]],
        )
