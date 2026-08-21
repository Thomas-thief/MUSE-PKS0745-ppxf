import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
import zap
from scipy.interpolate import interp1d

import os
from glob import glob

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from pathlib import Path
from urllib import request

from ppxf.ppxf import ppxf, robust_sigma
import ppxf.ppxf_util as util
import ppxf.sps_util as lib

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)

class Zapping():
    """
    Do a zapping of the cube to remove sky emission.
    """
    def do_zapping(cube_file, hdr, skycube, hdr_sky):
        w0 = hdr['CRVAL3']
        dw = hdr['CD3_3']    
        n_wave = hdr['NAXIS3']
        hdr['instrument'] = 'MUSE'
        wave_sci = w0 + np.arange(n_wave) * dw
        
        # load sky cube
        n_wave_sky = hdr_sky['NAXIS3']
        w0s = hdr_sky['CRVAL3']
        dws = hdr_sky['CD3_3']
        wave_sky = w0s + np.arange(n_wave_sky) * dws
        
        # interpolate
        if (n_wave) != (n_wave_sky):
            sky_interp = np.zeros((n_wave, skycube.shape[1], skycube.shape[2]), dtype=skycube.dtype)
            
            for i in range(skycube.shape[1]):
                for j in range(skycube.shape[2]):
                    f = interp1d(wave_sky, skycube[:, i, j], bounds_error=False, fill_value="extrapolate")
                    sky_interp[:, i, j] = f(wave_sci)

            outfile = "skycube_resampled.fits"
            hdus = [
                fits.PrimaryHDU(header=hdr_sky),
                fits.ImageHDU(data=sky_interp, header=hdr, name="DATA"),
                fits.ImageHDU(data=skycube, header=hdr, name="STAT")]
            fits.HDUList(hdus).writeto(outfile, overwrite=True)
            extSVD = zap.SVDoutput("skycube_resampled.fits") 

        else: 
            extSVD = zap.SVDoutput(cube_file) 

        cube_zap = cube_file.replace(".fits", "_ZAP.fits")
        zobj = zap.process(cube_file, outcubefits=cube_zap, extSVD=extSVD,overwrite=True, interactive=True)
        # plot a spectrum extracted from the original cube
        plt.plot(zobj.cube[:,150:160,280:290].sum(axis=(1,2)), 'g',linewidth=2, alpha=1,label='Original spectra')
        plt.plot(zobj.cleancube[:,150:160,280:290].sum(axis=(1,2)), 'b',linewidth=1,alpha=0.5,label='Clean spectra')
        plt.legend()
        plt.savefig('zapping_check.pdf',dpi=200)
        plt.show()
        print('zapping done')
        return cube_zap
        
