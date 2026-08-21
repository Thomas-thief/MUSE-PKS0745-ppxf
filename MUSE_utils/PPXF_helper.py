import numpy as np
import math
import matplotlib.pyplot as plt

from pathlib import Path
from urllib import request

from time import perf_counter as clock

from mpdaf.obj import Cube

from astropy.io import fits
import astropy.io.fits as pyfits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from ppxf.ppxf import ppxf, robust_sigma
import ppxf.ppxf_util as util
import ppxf.sps_util as lib
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from plotbin.display_bins import display_bins
from plotbin.plot_velfield import plot_velfield


def replace_nan(data, value=0):
    """
    Replaces NaN values in a list with a specified value.

    Args:
        data (list): The input list.
        value: The value to replace NaN with (default is 0).

    Returns:
        list: A new list with NaN values replaced.
    """
    new_data = [value if isinstance(x, float) and math.isnan(x) else x for x in data]
    return new_data




def _wave_convert(lam):
    """
    Convert between vacuum and air wavelengths using
    equation (1) of Ciddor 1996, Applied Optics 35, 1566
        http://doi.org/10.1364/AO.35.001566

    :param lam - Wavelength in Angstroms
    :return: conversion factor

    """
    lam = np.asarray(lam)
    sigma2 = (1e4/lam)**2
    fact = 1 + 5.792105e-2/(238.0185 - sigma2) + 1.67917e-3/(57.362 - sigma2)

    return fact


def emission_lines(ln_lam_temp, lam_range_gal, FWHM_gal, pixel=True,
                   tie_balmer=False, limit_doublets=False, vacuum=False):
    """
    ln_lam_temp, lam_range_gal, FWHM_gal, pixel=True,
                   tie_balmer=False, limit_doublets=False, vacuum=False
    Generates an array of Gaussian emission lines to be used as gas templates in pPXF.

    ****************************************************************************

    **ADDITIONAL LINES CAN BE ADDED BY EDITING THE CODE OF THIS PROCEDURE, WHICH 
    IS MEANT AS A TEMPLATE TO BE COPIED AND MODIFIED BY THE USERS AS NEEDED**

    ****************************************************************************

    Generally, these templates represent the instrumental line spread function
    (LSF) at the set of wavelengths of each emission line. In this case, pPXF
    will return the intrinsic (i.e. astrophysical) dispersion of the gas lines.

    Alternatively, one can input FWHM_gal=0, in which case the emission lines
    are delta-functions and pPXF will return a dispersion which includes both
    the instrumental and the intrinsic dispersion.

    For accuracy the Gaussians are integrated over the pixels boundaries.
    This can be changed by setting `pixel`=False.

    The [OI], [OIII] and [NII] doublets are fixed at theoretical flux ratio~3.

    The [OII] and [SII] doublets can be restricted to physical range of ratios.

    The Balmer Series can be fixed to the theoretically predicted decrement.

    Parameters
    ----------
    ln_lam_temp: array_like
        is the natural log of the wavelength of the templates in Angstrom.
        ``ln_lam_temp`` should be the same as that of the stellar templates.
    lam_range_gal: array_like
        is the estimated rest-frame fitted wavelength range. Typically::

            lam_range_gal = np.array([np.min(wave), np.max(wave)])/(1 + z),

        where wave is the observed wavelength of the fitted galaxy pixels and
        z is an initial rough estimate of the galaxy redshift.
    FWHM_gal: float, func or dict
        Instrumental resolution FWHM of the galaxy spectrum under study in
        Angstrom. One can pass either:
            * A scalar; 
            * The name "func" of a function ``func(wave)`` which returns the
              FWHM for a given vector of input wavelengths in Angstrom; 
            * A dictionary ``{"lam":lam, "fwhm":fwhm}`` with the wavelength and
              corresponding instrumental resolution of every pixel of the
              galaxy spectrum in Angstroms.
    pixel: bool, optional
        Set this to ``False`` to ignore pixels integration (default ``True``).
    tie_balmer: bool, optional
        Set this to ``True`` to tie the Balmer lines according to a theoretical
        decrement (case B recombination T=1e4 K, n=100 cm^-3).

        IMPORTANT: The relative fluxes of the Balmer components assumes the
        input spectrum has units proportional to ``erg/(cm**2 s A)``.
    limit_doublets: bool, optional
        Set this to True to limit the ratio of the [OII] and [SII] doublets to
        the ranges allowed by atomic physics.

        An alternative to this keyword is to use the ``constr_templ`` keyword
        of pPXF to constrain the ratio of two templates weights.

        IMPORTANT: when using this keyword, the two output fluxes (flux_1 and
        flux_2) provided by pPXF for the two lines of the doublet, do *not*
        represent the actual fluxes of the two lines, but the fluxes of the two
        input *doublets* of which the fit is a linear combination.
        If the two doublets templates have line ratios rat_1 and rat_2, and
        pPXF prints fluxes flux_1 and flux_2, the actual ratio and flux of the
        fitted doublet will be::

            flux_total = flux_1 + flux_1
            ratio_fit = (rat_1*flux_1 + rat_2*flux_2)/flux_total

        EXAMPLE: For the [SII] doublet, the adopted ratios for the templates are::

            ratio_d1 = flux([SII]6716/6731) = 0.44
            ratio_d2 = flux([SII]6716/6731) = 1.43.

        When pPXF prints (and returns in pp.gas_flux)::

            flux([SII]6731_d1) = flux_1
            flux([SII]6731_d2) = flux_2

        the total flux and true lines ratio of the [SII] doublet are::

            flux_total = flux_1 + flux_2
            ratio_fit([SII]6716/6731) = (0.44*flux_1 + 1.43*flux_2)/flux_total

        Similarly, for [OII], the adopted ratios for the templates are::

            ratio_d1 = flux([OII]3729/3726) = 0.28
            ratio_d2 = flux([OII]3729/3726) = 1.47.

        When pPXF prints (and returns in pp.gas_flux)::

            flux([OII]3726_d1) = flux_1
            flux([OII]3726_d2) = flux_2

        the total flux and true lines ratio of the [OII] doublet are::

            flux_total = flux_1 + flux_2
            ratio_fit([OII]3729/3726) = (0.28*flux_1 + 1.47*flux_2)/flux_total

    vacuum:  bool, optional
        set to ``True`` to assume wavelengths are given in vacuum.
        By default the wavelengths are assumed to be measured in air.

    Returns
    -------
    emission_lines: ndarray
        Array of dimensions ``[ln_lam_temp.size, line_wave.size]`` containing
        the gas templates, one per array column.

    line_names: ndarray
        Array of strings with the name of each line, or group of lines'

    line_wave: ndarray
        Central wavelength of the lines, one for each gas template'

    """

    if isinstance(FWHM_gal, dict):
        FWHM_gal1 = lambda lam: np.interp(lam, FWHM_gal["lam"], FWHM_gal["fwhm"])
    else:
        FWHM_gal1 = FWHM_gal

    #        Balmer:     H10       H9         H8        Heps    Hdelta    Hgamma    Hbeta     Halpha
    balmer = np.array([3798.983, 3836.479, 3890.158, 3971.202, 4102.899, 4341.691, 4862.691, 6564.632])  # vacuum wavelengths

    if tie_balmer:

        # Balmer decrement for Case B recombination (T=1e4 K, ne=100 cm^-3)
        # from Storey & Hummer (1995) https://ui.adsabs.harvard.edu/abs/1995MNRAS.272...41S
        # In electronic form https://cdsarc.u-strasbg.fr/viz-bin/Cat?VI/64
        # See Table B.7 of Dopita & Sutherland (2003) https://www.amazon.com/dp/3540433627
        # Also see Table 4.2 of Osterbrock & Ferland (2006) https://www.amazon.co.uk/dp/1891389343/
        wave = balmer
        if not vacuum:
            wave = util.vac_to_air(wave)
        gauss = util.gaussian(ln_lam_temp, wave, FWHM_gal1, pixel)
        ratios = np.array([0.0530, 0.0731, 0.105, 0.159, 0.259, 0.468, 1, 2.86])
        # Account for varying log-sampled pixel size in Angstrom
        ratios *= wave[-2]/wave
        emission_lines = gauss @ ratios
        line_names = ['Balmer']
        w = (lam_range_gal[0] < wave) & (wave < lam_range_gal[1])
        line_wave = np.mean(wave[w]) if np.any(w) else np.mean(wave)

    else:

        line_wave = balmer
        if not vacuum:
            line_wave = util.vac_to_air(line_wave)
        line_names = ['H10', 'H9', 'H8', 'Heps', 'Hdelta', 'Hgamma', 'Hbeta', 'Halpha']
        emission_lines = util.gaussian(ln_lam_temp, line_wave, FWHM_gal1, pixel)

    if limit_doublets:

        # The line ratio of this doublet lam3727/lam3729 is constrained by
        # atomic physics to lie in the range 0.28--1.47 (e.g. fig.5.8 of
        # Osterbrock & Ferland (2006) https://www.amazon.co.uk/dp/1891389343/).
        # We model this doublet as a linear combination of two doublets with the
        # maximum and minimum ratios, to limit the ratio to the desired range.
        #       -----[OII]-----
        wave = [3727.092, 3729.875]    # vacuum wavelengths
        if not vacuum:
            wave = util.vac_to_air(wave)
        names = ['[OII]3726_d1', '[OII]3726_d2']
        gauss = util.gaussian(ln_lam_temp, wave, FWHM_gal1, pixel)
        doublets = gauss @ [[1, 1], [0.28, 1.47]]  # produces *two* doublets
        emission_lines = np.column_stack([emission_lines, doublets])
        line_names = np.append(line_names, names)
        line_wave = np.append(line_wave, wave)

        # The line ratio of this doublet lam6717/lam6731 is constrained by
        # atomic physics to lie in the range 0.44--1.43 (e.g. fig.5.8 of
        # Osterbrock & Ferland (2006) https://www.amazon.co.uk/dp/1891389343/).
        # We model this doublet as a linear combination of two doublets with the
        # maximum and minimum ratios, to limit the ratio to the desired range.
        #        -----[SII]-----
        wave = [6718.294, 6732.674]    # vacuum wavelengths
        if not vacuum:
            wave = util.vac_to_air(wave)
        names = ['[SII]6731_d1', '[SII]6731_d2']
        gauss = util.gaussian(ln_lam_temp, wave, FWHM_gal1, pixel)
        doublets = gauss @ [[0.44, 1.43], [1, 1]]  # produces *two* doublets
        emission_lines = np.column_stack([emission_lines, doublets])
        line_names = np.append(line_names, names)
        line_wave = np.append(line_wave, wave)

    else:

        # Here the two doublets are free to have any ratio
        #         -----[OII]-----     -----[SII]-----
        wave = [3727.092, 3729.875, 6718.294, 6732.674]  # vacuum wavelengths
        if not vacuum:
            wave = util.vac_to_air(wave)
        names = ['[OII]3726', '[OII]3729', '[SII]6716', '[SII]6731']
        gauss = util.gaussian(ln_lam_temp, wave, FWHM_gal1, pixel)
        emission_lines = np.column_stack([emission_lines, gauss])
        line_names = np.append(line_names, names)
        line_wave = np.append(line_wave, wave)

    # Here the lines are free to have any ratio ### #I Modified this part of the code  ###################################
    #       -----[NeIII]-----    HeII      HeI -[N II] λ5755 -  [OII] 7319 - [OII] 7330 - SIII_9069 - SIII_9532  - [Ar III]7135.8  - [O III] λ4363 - [Ca II] λ7291    - [Ca II] λ7323    - [N I] λ5198 - 5197.90 Å - [N I] λ5200 → 5200.26 Å

    wave = [6549.860, 6585.271,4960.295, 5008.240, 3968.59, 3869.86, 4687.015, 5877.243, 7319, 7330,9069 ,9532, 7135.8,5755, 4363 ,7291.47,7323.89,5197.90,5200.26,9014.909,8862.782,8891.910,9068.600]  # vacuum wavelengths
    if not vacuum:
        wave = util.vac_to_air(wave)
    names = ['[NII]6548','[NII]6584','[OIII]4959','[OIII]5007','[NeIII]3968', '[NeIII]3869', 'HeII4687', 'HeI5876','[OII]7319','[OII]7330','[SIII]9069','[SIII]9532','[ArIII]7135','[NII]5755','[OIII]4363','[CaII]7291','[CaII]7323','[NI]5198','[NI]5200','Pa10','Pa11','[FeII]8891','[SIII]9068']
    
    gauss = util.gaussian(ln_lam_temp, wave, FWHM_gal1, pixel)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, wave)

    ######### Doublets with fixed ratios #########

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #        -----[OIII]-----
    #wave = [4960.295, 5008.240]    # vacuum wavelengths
    #if not vacuum:
     #   wave = util.vac_to_air(wave)
    #doublet = util.gaussian(ln_lam_temp, wave, FWHM_gal1, pixel) @ [0.33, 1]
    #emission_lines = np.column_stack([emission_lines, doublet])
    # single template for this doublet
    #line_names = np.append(line_names, '[OIII]5007_d')
    #line_wave = np.append(line_wave, wave[1])

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #        -----[OI]-----
    wave = [6302.040, 6365.535]    # vacuum wavelengths
    if not vacuum:
        wave = util.vac_to_air(wave)
    doublet = util.gaussian(ln_lam_temp, wave, FWHM_gal1, pixel) @ [1, 0.33]
    emission_lines = np.column_stack([emission_lines, doublet])
    # single template for this doublet
    line_names = np.append(line_names, '[OI]6300_d')
    line_wave = np.append(line_wave, wave[0])

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #       -----[NII]-----
    #wave = [6549.860, 6585.271]    # air wavelengths
    #if not vacuum:
    #    wave = util.vac_to_air(wave)
    #doublet = util.gaussian(ln_lam_temp, wave, FWHM_gal1, pixel) @ [0.33, 1]
    #emission_lines = np.column_stack([emission_lines, doublet])

    # single template for this doublet
    #line_names = np.append(line_names, '[NII]6583_d')
    #line_wave = np.append(line_wave, wave[1])

    # Only include lines falling within the estimated fitted wavelength range.
    #
    w = (lam_range_gal[0] < line_wave) & (line_wave < lam_range_gal[1])
    emission_lines = emission_lines[:, w]
    line_names = line_names[w]
    line_wave = line_wave[w]

    print('Emission lines included in gas templates:')
    print(line_names)

    return emission_lines, line_names, line_wave


def determine_mask(ln_lam, lam_range_temp, redshift=0, width=800):
    """
    Generates a mask to avoid fitting the region possibly contaminated by a
    given set of gas emission lines. This is meant to be used as input for pPXF.
    Here ``mask = True`` for the pixels to include in the fit.

    :param ln_lam: Natural logarithm np.log(wave) of the wavelength in
        Angstrom of each pixel of the log rebinned *galaxy* spectrum.
    :param lam_range_temp: Two elements vectors [lam_min_temp, lam_max_temp]
        with the minimum and maximum wavelength in Angstrom in the stellar
        *template* used in pPXF.
    :param z: Estimate of the galaxy redshift.
    :return: boolean vector mask to be used as input for pPXF

    """
     #   wave = [3968.59, 3869.86, 4687.015, 5877.243, 7319, 7330,9069 ,9532, 7135.8,5755, 4363 ,7291.47,7323.89,5197.90,5200.26]  # vacuum wavelengths
    #  -----[OII]-----    Hdelta   Hgamma   Hbeta   -----[OIII]-----   [OI]    -----[NII]-----   Halpha   -----[SII]-----
    lines = np.array([3726.03, 3728.82, 4101.76, 4340.47, 4861.33, 4958.92, 5006.84, 6300.30, 6363.78, 6548.03, 6583.41, 6562.80, 6716.47, 6730.85,7291.47,7323.89,5197.90,5200.26])
    # width/2 of masked gas emission region in km/s
    dv = np.full_like(lines, width)
    c = 299792.458  # speed of light in km/s

    flag = False
    for line, dvj in zip(lines, dv):
        flag |= (ln_lam > np.log(line*(1 + redshift)) - dvj/c) \
            & (ln_lam < np.log(line*(1 + redshift)) + dvj/c)

    # Mask edges of stellar library
    flag |= ln_lam > np.log(lam_range_temp[1]*(1 + redshift)) - 900/c
    flag |= ln_lam < np.log(lam_range_temp[0]*(1 + redshift)) + 900/c

    return ~flag

def replace_nan(data, value=0):
    """
    Replaces NaN values in a list with a specified value.

    Args:
        data (list): The input list.
        value: The value to replace NaN with (default is 0).

    Returns:
        list: A new list with NaN values replaced.
    """
    new_data = [value if isinstance(x, float) and math.isnan(x) else x for x in data]
    return new_data

class read_data_cube:
    def __init__(self, filename, lam_range, redshift):
        """Read data cube, de-redshift, log rebin and compute coordinates of each spaxel."""

        self.read_fits_file(filename)

        # Only use the specified rest-frame wavelength range
        wave = self.wave/(1 + redshift)      # de-redshift the spectrum
        w = (wave > lam_range[0]) & (wave < lam_range[1])
        wave = wave[w]
        cube = self.cube[w, ...]
        cubevar = self.cubevar[w, ...]

        signal = np.nanmedian(cube, 0)
        noise = np.sqrt(np.nanmedian(cubevar, 0))

        # Create coordinates centred on the brightest spaxel
        jm = np.argmax(signal)
        row, col = map(np.ravel, np.indices(cube.shape[-2:]))
        x = (col - col[jm])*self.pixsize
        y = (row - row[jm])*self.pixsize

        # Transform cube into 2-dim array of spectra
        npix = cube.shape[0]
        spectra = cube.reshape(npix, -1)        # create array of spectra [npix, nx*ny]
        variance = cubevar.reshape(npix, -1)    # create array of variance [npix, nx*ny]
        #print(f"despues del reshape{spectra.shape}--{variance.shape}")
        c = 299792.458  # speed of light in km/s
        velscale = np.min(c*np.diff(np.log(wave)))  # Preserve smallest velocity step
        lam_range_temp = np.array([np.min(wave), np.max(wave)])
        spectra, ln_lam_gal, velscale = util.log_rebin(lam_range_temp, spectra, velscale=velscale)
        variance, _, _ = util.log_rebin(lam_range_temp, variance, velscale=velscale) #!!!!!!!!!!!
        #print(f"despues del log_rebin{spectra.shape}--{variance.shape}")
        # Coordinates and spectra only for spaxels with enough signal
        self.spectra = spectra
        self.variance = variance
        self.x = x#[jm]
        self.y = y#[jm]
        self.signal = signal.ravel()
        self.noise = noise.ravel()

        self.col = col + 1   # start counting from 1
        self.row = row + 1
        self.velscale = velscale
        self.ln_lam_gal = ln_lam_gal
        self.fwhm_gal = self.fwhm_gal/(1 + redshift)
        self.header = self.header

###############################################################################

    def read_fits_file(self, filename):
        """
        Read MUSE cube, noise, wavelength, spectral FWHM and pixel size.

        It must return the cube and cuberr as (npix, nx, ny) and wave as (npix,)

        IMPORTANT: This is not a general function! Its details depend on the
        way the data were stored in the FITS file and the available keywords in
        the FITS header. One may have to adapt the function to properly read
        the FITS file under analysis.                
        """
        hdu = fits.open(filename)
        header = hdu[1].header
        cube = hdu[1].data
        cubevar = hdu[2].data

        # Only use the specified rest-frame wavelength range
        wave = header['CRVAL3'] + header['CD3_3']*np.arange(cube.shape[0])

        self.cube = cube
        self.cubevar = cubevar
        self.wave = wave
        self.fwhm_gal = 2.62  # Median FWHM = 2.62Å. Range: 2.51--2.88 (ESO instrument manual).
        self.pixsize = header["CDELT2"]
        self.header = header

###############################################################################

# def clip_outliers(galaxy, bestfit, mask):
#     """
#     Repeat the fit after clipping bins deviants more than 3*sigma in relative
#     error until the bad bins don't change any more. This function uses eq.(34)
#     of Cappellari (2023) https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C
#     """
#     while True:
#         scale = galaxy[mask] @ bestfit[mask]/np.sum(bestfit[mask]**2)
#         resid = scale*bestfit[mask] - galaxy[mask]
#         err = robust_sigma(resid, zero=1)
#         ok_old = mask
#         mask = np.abs(bestfit - galaxy) < 6.0*err
#         if np.array_equal(mask, ok_old):
#             break
            
#     return mask

def clip_outliers(galaxy, bestfit, mask, max_iter=10):
    """
    Repeat the fit after clipping bins deviants more than 3*sigma in relative
    error until the bad bins don't change any more. This function uses eq.(34)
    of Cappellari (2023) https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C
    """
    for i in range(max_iter):
        scale = galaxy[mask] @ bestfit[mask] / np.sum(bestfit[mask]**2)
        resid = scale*bestfit[mask] - galaxy[mask]
        err = robust_sigma(resid, zero=1)
        ok_old = mask
        mask = np.abs(bestfit - galaxy) < 3.0*err # 6.0
        if np.array_equal(mask, ok_old):
            break
    else:
        print(f"clip_outliers: did not converge after {max_iter} iterations, "
              f"using last mask (possible oscillation)")

    return mask


def ppxf_fit_and_clean(templates, galaxy, noise, velscale, start, mask0, lam, lam_temp, plot=True, quiet=False):
    """
    This is a simple pPXF wrapper. It perform two pPXF fits: the first one
    serves to estimate the scatter in the spectrum and identify the outlier
    pixels. The second fit uses the mask obtained from the first fit to exclude
    the outliers. The general approach used in this function is described in
    Sec.6.5 of Cappellari (2023) https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C
    """
    mask = mask0.copy()
    pp = ppxf(templates, galaxy, noise, velscale, start,
              moments=4, degree=10, mdegree=-1, lam=lam, lam_temp=lam_temp,
              mask=mask, quiet=quiet,clean=False)

    if plot:
        plt.figure(figsize=(20, 3))
        plt.subplot(121)
        pp.plot()
        plt.title("Initial pPXF fit before outliers removal")

    mask = clip_outliers(galaxy, pp.bestfit, mask)

    # Add clipped pixels to the original masked emission lines regions and repeat the fit
    mask &= mask0
    pp = ppxf(templates, galaxy, noise, velscale, start,
              moments=4, degree=10, mdegree=-1, lam=lam, lam_temp=lam_temp,
              mask=mask, quiet=quiet, clean=True)
    
    #pp.optimal_template = templates.reshape(templates.shape[0], -1) @ pp.weights

    # resid = (pp.galaxy - pp.bestfit)[pp.goodpixels]
    # pp.sn = np.nanmedian(pp.galaxy[pp.goodpixels])/robust_sigma(resid)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

    velbin, sigbin, h3_a, h4_a = pp.sol
    start = [velbin, sigbin]

    pp = ppxf(templates, galaxy, noise, velscale, start, fixed = [True,True],
                moments=2, degree=-1, mdegree=8, lam=lam, lam_temp=lam_temp,
                mask=mask, quiet=quiet,clean=False)

    pp.optimal_template = templates.reshape(templates.shape[0], -1) @ pp.weights
    pp.sol = [velbin, sigbin, h3_a, h4_a]
    resid = (pp.galaxy - pp.bestfit)[pp.goodpixels]
    pp.sn = np.nanmedian(pp.galaxy[pp.goodpixels])/robust_sigma(resid)
    # pp se saca la cinematica y se inserta otro ajuste que esta en discord !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if plot:
        plt.subplot(122)
        pp.plot()
    
    return pp

def cut_cube_simple(cube, center, size):
    '''
    Function to cut a fits cube using mpdaf
    cube : path to the fits cube
    center : [RA, DEC] in degrees
    size : size of the cut in arcsec
    returns : mpdaf Cube object
    '''
    c = SkyCoord(center[0]*u.deg, center[1]*u.deg, frame='icrs')
    cube = Cube(cube)
    cut = cube.subcube(center=c, size=size * u.arcsec)
    return cut

def cut_cube(filename,ra,dec,size,filename_output):
    '''
    filename_output : path to the fits cube output
    filename: path to the fits cube input
    ra,dec:  RA in h:m:s, d:m:s
    size = () in pixels
    
    '''
    # Cut the cube to only fit the selected position 
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    from astropy.io import fits

    hdu = fits.open(filename)
    hdu.info()
    head0=hdu[0].header
    head = hdu[1].header
    head['CRVAL3'] = 4749.6611328125
    head['CDELT3'] = 1.25
    head['CDELT1'] = 5.55555555555556E-05 #head['CD1_1']

    cube = hdu[1].data
    cube_std = hdu[2].data
    z = cube.shape[0]

    # Delete the third axis
    wcs_original = WCS(hdu[1].header)
    wcs = wcs_original.dropaxis(2)
    print(wcs)
    
    naxis = cube.shape[0] # yaxis sieze 
    cube_cut = []
    cube_std_cut = []
    for i in np.arange(0,z):
        data = cube[i,:,:]
        data_var = cube_std[i,:,:]

        # Cut the image #########################
        shape = size
        position = SkyCoord(ra+dec, frame='icrs')
        cutout = Cutout2D(data, position, shape, wcs=wcs)
        cube_cut.append(cutout.data)
    
        cutout2 = Cutout2D(data_var, position, shape, wcs=wcs)
        cube_std_cut.append(cutout2.data)

    # Save the image 
    hdu = fits.HDUList()
    hdu.append(fits.PrimaryHDU(header=head0))

    hdu.header = head
    hdu.header.update(cutout.wcs.to_header())
    hdu.header['CDELT3'] =  head['CDELT3'] 
    hdu.header['CTYPE3'] =  head['CTYPE3'] 
    hdu.header['CRVAL3'] =  head['CRVAL3'] 
    hdu.header['CRPIX3'] =  head['CRPIX3'] 
    hdu.header['CD3_3'] =  head['CD3_3'] 
    hdu.header['CUNIT3'] =  head['CUNIT3'] 
    hdu.header['CRDER3'] =  head['CRDER3']     

    hdu.append(fits.ImageHDU(cube_cut, header=hdu.header, name='DATA'))
    hdu.append(fits.ImageHDU(cube_std_cut, header=hdu.header, name='VARIANCE'))
    hdu.writeto(filename_output, overwrite=True,output_verify='silentfix+ignore')