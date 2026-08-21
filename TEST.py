import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import astropy.units as u

# import MUSE_utils.PPXF_astrometry_correction as ppxf_ac
import MUSE_utils.PPXF_class as ppxf_c
# import MUSE_utils.PPXF_extinction_correction as ppxf_ec
# import MUSE_utils.PPXF_fitting as ppxf_f
# import MUSE_utils.PPXF_helper as ppxf_h
# import MUSE_utils.PPXF_redshift_calculator as ppxf_rc
# import MUSE_utils.PPXF_zapping as ppxf_z

from mpdaf.obj import Cube, WCS, WaveCoord

name_cube = "PKS0745_cube_wcscorr_ext-cor_small.fits"
hdu = fits.open(name_cube)
header_cube = hdu[0].header
header_data = hdu[1].header
crop_data = hdu[1].data # spectroXpixXpix
desv_data = hdu[2].data # spectroXpixXpix

vvcs= WCS(hdr=header_data)
wave1 = WaveCoord(cdelt=1.25, crval=4750.287109375, cunit=u.angstrom)
ESE = Cube(filename=name_cube, wave=wave1, data=crop_data, var=desv_data, primary_header=header_cube, wcs=vvcs)
target_sn = 3 #np.sum(crop_data) / np.sqrt(np.sum(desv_data**2))
#preguntar como lo calculo o estimo?, se tiene que revisar bien con la edad de poblacion segun la autoa
#b_unit = 10**(-20)*u.erg/u.s/u.cm**2/u.angstrom
# PPFX_class
cinematica_stellar = ppxf_c.stellar_kinematics(ESE, target_sn, desv_data, crop_data)
cinematica_stellar.compute_voronoi()

# plt.imshow(np.nanmean(hdu[1].data,axis=0))
# plt.show()