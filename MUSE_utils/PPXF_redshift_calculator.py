import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
#import pyregion
import math
import pandas as pd

import os
from glob import glob

#from tqdm import tqdm

from mpdaf.obj import Cube, WaveCoord
from mpdaf.drs import PixTable

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from pathlib import Path
from urllib import request

from ppxf.ppxf import ppxf, robust_sigma
import ppxf.ppxf_util as util
import ppxf.sps_util as lib

from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


##########################################################################
# CLASS 1 - Data cube reader (your original working code)
##########################################################################

class ReadDataSpectrum:
        def __init__(self, filename, lam_range, redshift):
            self.filename = filename
            self.redshift = redshift
            self.lam_range = lam_range

            hdu = fits.open(filename)
            head = hdu[1].header
            cube = hdu[1].data  
        #   cubevar = hdu[2].data 

            # Only use the specified rest-frame wavelength range
            wave = head['CRVAL1'] + head['CDELT1']*np.arange(cube.shape[0])

            self.cube = cube
        #   self.cubevar = cubevar
            self.wave = wave
            self.fwhm_gal = 2.62  # Median FWHM = 2.62Å. Range: 2.51--2.88 (ESO instrument manual). 
            self.pixsize = 0.2

            # wavelength selection in rest-frame
            wave = self.wave / (1 + redshift)
            w = (wave > lam_range[0]) & (wave < lam_range[1])
            wave = wave[w]
            cube = self.cube[w, ...]

            signal = cube

            # brightest spaxel
            jm = np.argmax(signal)

            # log-rebinning
            c = 299792.458
            velscale = np.min(c * np.diff(np.log(wave)))
            lam_range_temp = [np.min(wave), np.max(wave)]

            spectra, ln_lam_gal, velscale = util.log_rebin(
                lam_range_temp, cube, velscale=velscale
            )

            self.spectra = spectra
            self.signal = signal.ravel()
            self.velscale = velscale
            self.ln_lam_gal = ln_lam_gal
            self.fwhm_gal = self.fwhm_gal / (1 + redshift)

##########################################################################
# CLASS 2 — Utilities (your functions, now static methods)
##########################################################################

class CubeUtils:

    @staticmethod
    def replace_nan(data, value=0):
        return [value if isinstance(x, float) and math.isnan(x) else x for x in data]

    @staticmethod
    def find_max_index(image_array):
        max_index_flat = np.argmax(image_array)
        max_index_2d = np.unravel_index(max_index_flat, image_array.shape)
        maximum = np.max(image_array)
        return max_index_2d, maximum

    # --------------------------------------------------------------
    @staticmethod
    def subtract_peak_spectrum(z, filename):
        """Your original working extraction method intact."""
        cube = Cube(filename, ext=1)
        header = cube.data_header
        wcs = WCS(header, naxis=2)

        dw = header['CD3_3']
        w_min = header['CRVAL3']
        nPixels = header['NAXIS3']
        wavelength = np.linspace(w_min, w_min + dw * nPixels, nPixels, endpoint=False)

        image = cube.get_band_image('Johnson_V')
        max_index, val = CubeUtils.find_max_index(image.data)

        coor_x = max_index[1] + 1
        coor_y = max_index[0] + 1

        world = wcs.wcs_pix2world(coor_x, coor_y, 1)
        subcube = cube.subcube_circle_aperture(
            center=(float(world[1]), float(world[0])),
            radius=1
        )

        spec = subcube.sum(axis=(1, 2))

        # Plot identical to your original
        f = plt.figure(figsize=(15, 5))
        f.add_subplot(1, 2, 1)
        vmin0, vmax0 = np.percentile(image.data[~np.isnan(image.data)], (1, 99.5))
        plt.imshow(image.data, origin='lower', cmap='inferno', vmin=vmin0, vmax=vmax0)
        circ = plt.Circle((coor_x - 1, coor_y - 1), radius=2.5,
                          linewidth=2, edgecolor='darkgoldenrod', fill=False)
        plt.gca().add_artist(circ)

        f.add_subplot(1, 2, 2)
        plt.plot(wavelength / (1 + z), spec.data)
        plt.tight_layout()
        plt.show()

        out = filename.replace(".fits", "").replace(".FITS", "") + "_SPEC_CENTER.fits"
        spec.write(out)
        print(f"Spectrum written to {out}")


##########################################################################
# CLASS 3 — Stellar kinematics (your original pPXF workflow)
##########################################################################

class StellarKinematics:
    def __init__(self, s):
        self.s = s
        self.signal = s.signal

    # --------------------------------------------------------------
    def clip_outliers(self, galaxy, bestfit, mask):
        while True:
            scale = galaxy[mask] @ bestfit[mask] / np.sum(bestfit[mask] ** 2)
            resid = scale * bestfit[mask] - galaxy[mask]
            err = robust_sigma(resid, zero=1)
            ok_old = mask
            mask = np.abs(bestfit - galaxy) < 3 * err
            if np.array_equal(mask, ok_old):
                break
        return mask

    # --------------------------------------------------------------
    def ppxf_fit_and_clean(self, templates, galaxy, noise,
                           velscale, start, mask0, lam, lam_temp,
                           plot=True, quiet=False):

        mask = mask0.copy()
        pp = ppxf(templates, galaxy, noise, velscale, start,
                  moments=2, degree=10, mdegree=-1,
                  lam=lam, lam_temp=lam_temp,
                  mask=mask, quiet=quiet)

        if plot:
            plt.figure(figsize=(20, 5))
            plt.subplot(121)
            pp.plot()
            plt.title("Initial fit")

        mask = self.clip_outliers(galaxy, pp.bestfit, mask)
        mask &= mask0

        pp = ppxf(templates, galaxy, noise, velscale, start,
                  moments=2, degree=10, mdegree=-1,
                  lam=lam, lam_temp=lam_temp,
                  mask=mask, quiet=quiet, clean=True)

        pp.optimal_template = templates.reshape(templates.shape[0], -1) @ pp.weights

        resid = (pp.galaxy - pp.bestfit)[pp.goodpixels]
        pp.sn = np.nanmedian(pp.galaxy[pp.goodpixels]) / robust_sigma(resid)

        if plot:
            plt.subplot(122)
            pp.plot()
            plt.show()
        return pp

    # --------------------------------------------------------------
    def setup_stellar_kinematics(self, start, redshift):
        """Your original working method — unchanged."""
        
        print("Setting up SPS templates…")

        sps_name = 'emiles'
        ppxf_dir = Path(lib.__file__).parent
        basename = f"spectra_{sps_name}_9.0.npz"
        filename = ppxf_dir / 'sps_models' / basename

        if not filename.is_file():
            url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
            request.urlretrieve(url, filename)

        s = self.s

        FWHM_gal = 2.62
        sps = lib.sps_lib(filename, s.velscale, FWHM_gal, norm_range=[5070, 5950])

        npix, *reg_dim = sps.templates.shape
        sps.templates = sps.templates.reshape(npix, -1)
        sps.templates /= np.median(sps.templates)

        lam_range_temp = np.exp(sps.ln_lam_temp[[0, -1]])
        mask0 = util.determine_mask(s.ln_lam_gal, lam_range_temp, width=1000)
        lam_gal = np.exp(s.ln_lam_gal)

        galaxy2 = s.spectra / np.nanmedian(s.spectra)
        noise = galaxy2 * 0.1

        pp = self.ppxf_fit_and_clean(
            sps.templates, galaxy2, noise,
            s.velscale, start, mask0,
            lam_gal, sps.lam_temp
        )

        # ---- redshift and error
        c = 299792.458
        errors = pp.error * np.sqrt(pp.chi2)
        redshift_fit = (1 + redshift) * np.exp(pp.sol[0] / c) - 1
        redshift_err = (1 + redshift_fit) * errors[0] / c

        print(f"Best-fitting redshift = {redshift_fit} ± {redshift_err}")
        return redshift_fit, redshift_err
