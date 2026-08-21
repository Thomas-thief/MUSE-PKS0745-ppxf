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


    
import numpy as np
from astropy.io import fits

class GalacticExtinctionCorrector:
    """
    Apply Galactic extinction correction to a MUSE (or general FITS) cube
    using the Schlafly & Finkbeiner (2011) extinction curve.
    """

    def __init__(self, EBV, Rv=3.1):
        """
        EBV : E(B-V) from dust maps
        Rv  : total-to-selective extinction ratio (default 3.1)
        """
        self.EBV = EBV
        self.Rv = Rv
        self.A_V = Rv * EBV   # always compute A_V once

    # -------------------------------------------------------------
    def gal_extinct(self, wave):
        """
        Galactic extinction A_lambda from Schlafly & Finkbeiner (2011)

        wave : array in Angstrom
        returns A_lambda in magnitudes
        """
        m = wave / 10000.0
        x = 1.0 / m
        y = x - 1.82

        ax = (1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 +
              0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 +
              0.32999*y**7)

        bx = (1.41338*y + 2.28305*y**2 + 1.07233*y**3 -
              5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 -
              2.09002*y**7)

        Arat = ax + bx / self.Rv
        return Arat * self.A_V

    # -------------------------------------------------------------
    def correct_cube(self, filename):
        """
        Applies extinction correction and writes a new corrected cube.

        filename : path to the FITS cube
        returns  : name of corrected cube
        """

        # --- read FITS cube ---
        hdu = fits.open(filename)
        header = hdu[1].header
        data = hdu[1].data
        error = hdu[2].data

        # wavelength solution
        n_wave = data.shape[0]
        wave = header["CRVAL3"] + np.arange(n_wave)*header["CD3_3"]

        # --- extinction A_lambda ---
        A_lambda = self.gal_extinct(wave)
        corr_factor = 10**(A_lambda[:, np.newaxis, np.newaxis] / -2.5)

        # apply correction
        data_corr  = data  / corr_factor
        error_corr = error / corr_factor

        # --- write new FITS ---
        new_hdul = fits.HDUList()
        new_hdul.append(fits.PrimaryHDU(header=hdu[0].header))
        new_hdul.append(fits.ImageHDU(data_corr,  header=hdu[1].header, name="DATA"))
        new_hdul.append(fits.ImageHDU(error_corr, header=hdu[2].header, name="VARIANCE"))

        outname = filename.replace(".fits", "_corr.fits")
        new_hdul.writeto(outname, overwrite=True)

        return outname





