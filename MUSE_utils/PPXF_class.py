from pathlib import Path
from urllib import request

from time import perf_counter as clock

import math

from astropy.io import fits
import astropy.io.fits as pyfits
from astropy.wcs import WCS

from ppxf.ppxf import ppxf, robust_sigma
import ppxf.ppxf_util as util
import ppxf.sps_util as lib
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from plotbin.display_bins import display_bins
from plotbin.plot_velfield import plot_velfield

from os import path
import fnmatch
import warnings
warnings.filterwarnings('ignore')

from collections import defaultdict
import os
import sys

from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from . import PPXF_helper as ph

c_kms = 299792.458

def robust_sigma(y, zero=False):
    """
    Biweight estimate of the scale (standard deviation).
    Implements the approach described in
    "Understanding Robust and Exploratory Data Analysis"
    Hoaglin, Mosteller, Tukey ed., 1983, Chapter 12B, pg. 417

    """
    y = np.ravel(y)
    d = y if zero else y - np.median(y)

    mad = np.median(np.abs(d))
    u2 = (d / (9.0 * mad)) ** 2  # c = 9
    good = u2 < 1.0
    u1 = 1.0 - u2[good]
    num = y.size * ((d[good] * u1**2) ** 2).sum()
    den = (u1 * (1.0 - 5.0 * u2[good])).sum()
    sigma = np.sqrt(num / (den * (den - 1.0)))  # see note in above reference

    return sigma



def save_fits(fitsfile_name,data,header):
    fits.writeto('MUSE_Maps/'+fitsfile_name+'.fits',data=data,header=header,overwrite=True)


class stellar_kinematics:
    def __init__(self,s,target_sn,noise,signal):
        self.s = s
        self.target_sn = target_sn
        self.noise = noise
        self.signal = signal

    def compute_voronoi(self, pixelsize=0.2, plot=True):
        """Run Voronoi binning using the stored data."""
        print("Running Voronoi binning...")

        self.bin_num = voronoi_2d_binning(
            self.s.x,           # x coordinates
            self.s.y,           # y coordinates
            self.signal,        # signal map
            self.noise,         # noise map
            self.target_sn,     # target S/N
            pixelsize=pixelsize,
            plot=plot,
            quiet=1
        )[0]


    def setup_stellar_kinematics(self):
        """Setup stellar templates using the stored data."""
        print("Setting up stellar templates...")
        s = self.s
        bin_num = self.bin_num
        # ## Setup stellar templates
        # pPXF can be used with any set of SPS population templates. However, I am currently providing (with permission) ready-to-use template files for four SPS. One can just uncomment one of the four models below. The included files are only a subset of the SPS that can be produced with the models, and one should use the relevant software/website to produce different sets of SPS templates if needed.
        # If you use the fsps v3.2 SPS model templates, please also cite in your paper Conroy et al. (2009) and Conroy et al. (2010).
        # If you use the GALAXEV v2020 SPS model templates, please also cite in your paper Bruzual & Charlot (2003).
        # If you use the E-MILES SPS model templates, please also cite in your paper Vazdekis et al. (2016). WARNING: The E-MILES models only include SPS with age > 63 Myr and are not recommended for highly star forming galaxies.
        # If you use the X-Shooter Spectral Library (XSL) SPS model templates, please also cite in your paper Verro et al. (2022). WARNING: The XSL models only include SPS with age > 50 Myr and are not recommended for highly star forming galaxies.

        sps_name = 'emiles'

        ppxf_dir = Path(lib.__file__).parent
        basename = f"spectra_{sps_name}_9.0.npz"
        filename = ppxf_dir / 'sps_models' / basename
        if not filename.is_file():
            url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
            request.urlretrieve(url, filename)

        FWHM_gal = None   # set this to None to skip templates broadening
        sps = lib.sps_lib(filename, s.velscale, FWHM_gal, norm_range=[5070, 5950])
        self.sps = sps

        npix, *reg_dim = sps.templates.shape
        sps.templates = sps.templates.reshape(npix, -1)
        sps.templates /= np.median(sps.templates) # Normalizes stellar templates by a scalar
        self.reg_dim = reg_dim


        lam_range_temp = np.exp(sps.ln_lam_temp[[0, -1]])
        #mask0 = util.determine_mask(s.ln_lam_gal, lam_range_temp, width=1000)
        mask0 = ph.determine_mask(s.ln_lam_gal, lam_range_temp, width=1000)

        # Define the region to mask, e.g. 6850–6950 Å
        lam_gal = np.exp(s.ln_lam_gal)
        self.lam_gal = lam_gal

        mask_region = (lam_gal > 7750) & (lam_gal < 7850)
        good_custom = ~mask_region   # True = allowed

        mask0 = mask0 & good_custom     # Logical AND → both must be valid

        # Option 2: set error to zero (pPXF will skip those pixels)
        goodpixels = np.where(~mask_region)[0]

        nbins = np.unique(bin_num).size
        velbin, sigbin, lg_age_bin, metalbin, nspax, h3, h4 = np.zeros((7, nbins))
        optimal_templates = np.empty((npix, nbins))

        for j in range(nbins):
            plot = True
            w = bin_num == j
            galaxy = np.nanmean(s.spectra[:, w], 1)
            galaxy = ph.replace_nan(galaxy,np.nanmean(galaxy))
            galaxy_normalized = galaxy/np.nanmedian(galaxy)     # Normalize spectrum to avoid numerical issues
            noise = (np.nanmean(s.noise))/np.nanmedian(galaxy_normalized)
            noise = np.full_like(galaxy_normalized,noise)

            pp_stars = ph.ppxf_fit_and_clean(sps.templates, galaxy_normalized, noise, s.velscale, s.start, mask0, lam_gal, sps.lam_temp, plot=plot, quiet=True)
            
            velbin[j], sigbin[j], h3[j], h4[j] = pp_stars.sol
            optimal_templates[:, j] = pp_stars.optimal_template

        self.optimal_templates = optimal_templates
        self.velbin = velbin
        self.sigbin = sigbin
        self.h3 = h3
        self.h4 = h4
        self.pp_stars = pp_stars

        light_weights = pp_stars.weights.reshape(reg_dim)
        lg_age_bin[j], metalbin[j] = sps.mean_age_metal(light_weights, quiet=not plot)

        if plot:
            txt = f"Voronoi bin {j + 1} / {nbins}; SPS: {sps_name}; $\\sigma$={sigbin[j]:.0f} km/s; S/N={pp_stars.sn:.1f}"
            print(txt + '\n' + '#'*78)
            plt.title(txt)
            plt.savefig(self.s.outfolder+'Stellar_fitting_example.png',dpi=300)
            plt.close()

        plt.subplots(1, 2, figsize=(10, 5))
        plt.subplots_adjust(wspace=0.5)

        plt.subplot(121)
        display_bins(s.x, s.y, bin_num, velbin, colorbar=1, label='V (km/s)',pixelsize=0.2)
        #plt.tricontour(s.x, s.y, -2.5*np.log10(signal/np.max(signal).ravel()), levels=np.arange(20));  # 1 mag contours

        plt.subplot(122)
        #sigbin = np.array(replace_nan(noise,np.nanmean(sigbin)))
        display_bins(s.x, s.y, bin_num, sigbin, colorbar=1, cmap='inferno', label='Vsigma (yr)',pixelsize=0.2)
        #plt.tricontour(s.x, s.y, -2.5*np.log10(signal/np.max(signal).ravel()), levels=np.arange(20));  # 1 mag contours

        plt.tight_layout()
        plt.show()

    def put_data_in_original_spaxels(self, bin_values):
        """
        Reconstruye un mapa 2D (ny, nx) desde valores binned (nbins,)
        y un bin_num que puede estar plano o 2D.
        """

        bin_values = np.asarray(bin_values)
        bin_num = np.asarray(self.bin_num)

        # Detectar dimensiones originales del cubo MUSE
        # s.spectra.shape = (nspectral, nspax)  cuando está aplanado
        # o (nspectral, ny, nx)
        if self.s.spectra.ndim == 3:
            ny, nx = self.s.spectra.shape[1], self.s.spectra.shape[2]
        else:
            # Si viene en forma plana, también recuperamos ny, nx del header
            ny = self.s.header["NAXIS2"]
            nx = self.s.header["NAXIS1"]

        # Asegurar que bin_num tenga forma 2D adecuada
        if bin_num.ndim == 1:
            bin_num = bin_num.reshape(ny, nx)

        # Crear mapa final
        full_map = np.zeros((ny, nx), dtype=float)

        # Llenar cada pixel con el valor de su Voronoi bin
        for b in range(bin_values.size):
            full_map[bin_num == b] = bin_values[b]

        return full_map


    def save_stellar_maps(self, prefix="stars_"):
        """
        Writes the stellar kinematic maps (V, sigma, h3, h4) to FITS files.
        """

        print("Saving stellar kinematic FITS maps...")

        # Reconstruct maps
        vel_map = self.put_data_in_original_spaxels(self.velbin)
        sig_map = self.put_data_in_original_spaxels(self.sigbin)
        h3_map  = self.put_data_in_original_spaxels(self.h3)
        h4_map  = self.put_data_in_original_spaxels(self.h4)

        # Save
        fits.writeto(self.s.outfolder + prefix + "vel.fits", vel_map, header=self.s.header, overwrite=True)
        fits.writeto(self.s.outfolder + prefix + "sigma.fits", sig_map, header=self.s.header, overwrite=True)
        fits.writeto(self.s.outfolder + prefix + "h3.fits",  h3_map,  header=self.s.header, overwrite=True)
        fits.writeto(self.s.outfolder + prefix + "h4.fits",  h4_map,  header=self.s.header, overwrite=True)


    def stellar_kinematics(self):
            self.compute_voronoi()
            self.setup_stellar_kinematics()
            self.save_stellar_maps()

            return self




def make_mask(lamdas, wavelengths):

    """
    Make a boolean mask which is False between each pair of wavelengths and True outside them.
    This is useful for masking skylines in our spectra

    Arguments:
        lamdas (array): An array of wavelength values
        wavelengths (list): A 2 component vector of low lambda and high lambda values we want to mask between
    Returns:
        (boolean array): A boolean of array of True outside the pair of wavelengths and False between them.
    """

    mask = np.ones_like(lamdas, dtype=bool)
    for pair in wavelengths:
        low, high = pair
        mask = mask & (lamdas < low) | (lamdas > high)

    return mask


def find_pixel_with_high_flux(filename,wave_range):

        hdu = pyfits.open(filename)
        hdu.info()
        head = hdu[0].header

        cube = hdu[0].data
        cube_std = hdu[1].data
        ny = cube.shape[1]
        nx = cube.shape[2]

        # Transform cube into 2-dim array of spectra
        npix = cube.shape[0]
        spectra = cube.reshape(npix, -1) # create array of spectra [npix, nx*ny]
        spectra_std = cube_std.reshape(npix, -1) # create array of spectra [npix, nx*ny]

        # Only use a restricted wavelength range
        wave = head['CRVAL3'] + head['CDELT3']*np.arange(npix)
        pixsize = abs(head["CDELT1"])*3600    # 0.2"

        w = (wave > wave_range[0]) & (wave < wave_range[1])
        spectra = spectra[w, :]
        spectra_std = spectra_std[w, :]
        wave = wave[w]
        C=3e6
        velscale = C*np.diff(np.log(wave[-2:]))  # Smallest velocity step
        lam_range_temp = [np.min(wave), np.max(wave)]
        spectra_std, ln_lam_gal, velscale = util.log_rebin(lam_range_temp, spectra_std, velscale=velscale)
        spectra, ln_lam_gal, velscale = util.log_rebin(lam_range_temp, spectra, velscale=velscale)
        lam_gal = np.exp(ln_lam_gal)

        # Mask tellutic emission #####################################################################
        telluric_lam_0=np.array([[6290, 6310]])
        telluric_lam_1=np.array([[6862, 6952]])
        telluric_lam_3=np.array([[6357, 6370]])

        masked_wavelengths=np.vstack([telluric_lam_1, telluric_lam_0, telluric_lam_3]).reshape(-1, 1, 2)
        string_masked_wavelengths=["{} to {}".format(pair[0][0], pair[0][1]) for pair in masked_wavelengths]
        
        pixel_mask=np.ones_like(lam_gal, dtype=bool)


        signal = np.median(spectra, 0)
        noise = np.sqrt(signal)

        nx ,ny =  nx, ny 
        x_array = np.arange(0,nx)
        y_array = np.arange(0,ny)
        spectrum_array = np.arange(0,len(signal))
    #    spectrum_array = spectrum_array.reshape(ny, nx)

        

        # find pixels with a high meadian value > pixels with a high Halpha emission
        maximum_val = np.where(np.isclose(signal, np.nanmax(signal)))

        # Find more values with large Halpha flux
        i_to_fit_2gaussian = np.where(signal>np.nanmax(signal)-10)

        #for j in i_to_fit_2gaussian:#np.arange(0,len(s.spectra)):
         #   plt.plot(s.spectra[:, j])
         #   plt.show()
        return i_to_fit_2gaussian



# ### 2. Two velocity components for the gas
#===========================================
#                   GAS KINEMATICS FITTING CLASS
# ===================================================================
class GasKinematicsFitter:
    

    def __init__(self, s, sps, lam_gal, velbin, sigbin, h3, h4,
                 bin_num, optimal_templates, fwhm_gal=1.5):
        """
        Parameters
        ----------
        s : PHANGS cube object with .spectra and .variance
        sps : stellar population library
        lam_gal : wavelength array
        velbin, sigbin, h3, h4 : stellar kinematic maps (per Voronoi bin)
        bin_num : Voronoi bin assignments for each spaxel
        optimal_templates : stellar templates per bin
        fwhm_gal : instrumental FWHM in Angstroms (MUSE: 1.5)
        """

        self.s = s
        self.sps = sps
        self.lam_gal = lam_gal
        self.velbin = velbin
        self.sigbin = sigbin
        self.h3 = h3
        self.h4 = h4
        self.bin_num = bin_num
        self.optimal_templates = optimal_templates
        self.fwhm_gal = fwhm_gal

        self.lam_range_gal = [np.min(lam_gal), np.max(lam_gal)]

        # Build templates immediately
        self._build_templates()


    # ------------------------------------------------------------------
    # Build single- and double-component emission-line templates
    # ------------------------------------------------------------------
    def _build_templates(self):
        """
        Create:
        - gas_templates_1, gas_names_1  (one kinematic component)
        - gas_templates_2, gas_names_2  (two components)
        """

        gas_temp_base, gas_names_base, line_wave = ph.emission_lines(
            self.sps.ln_lam_temp, self.lam_range_gal, self.fwhm_gal
        )

        self.gas_templates_base = np.asarray(gas_temp_base, float)
        self.gas_names_base = np.asarray(gas_names_base)
        self.line_wave = np.asarray(line_wave)
        n_lines = len(self.gas_names_base)

        # 1-comp
        self.gas_templates_1 = self.gas_templates_base.copy()
        self.gas_names_1 = np.array([f"{name}_(1)" for name in self.gas_names_base])

        # 2-comp
        gas_templates_2 = np.tile(self.gas_templates_base, 2)
        gas_names_2 = (
            [f"{name}_(1)" for name in self.gas_names_base] +
            [f"{name}_(2)" for name in self.gas_names_base]
        )
        self.gas_templates_2 = np.asarray(gas_templates_2)
        self.gas_names_2 = np.asarray(gas_names_2)


    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def _get_BIC(self, pp):
        chi2 = pp.chi2 * pp.dof
        k = pp.npix - pp.dof
        N = pp.npix
        return chi2 + k * np.log(N)

    def _get_Halpha_index(self, names, tag="Halpha_(1)"):
        idx = np.where(names == tag)[0]
        return idx[0] if len(idx) > 0 else None


    # ------------------------------------------------------------------
    # Fit a single spaxel
    # ------------------------------------------------------------------
    def _fit_single_spaxel(self, j, sn_min=40, dBIC_min=5, frac_min=0.1):

        # Extract spectrum + noise
        galaxy = np.asarray(self.s.spectra[:, j], float)
        galaxy = np.asarray(ph.replace_nan(galaxy, np.nanmean(galaxy)), float)

        var = np.asarray(self.s.variance[:, j], float)
        noise = np.sqrt(np.abs(var))
        noise = np.asarray(ph.replace_nan(noise, np.nanmean(noise)), float)



        # Stellar template from its Voronoi bin
        kbin = self.bin_num[j]
        template = np.asarray(self.optimal_templates[:, kbin], float)

        lam_gal = self.lam_gal
        lam_temp = self.sps.lam_temp

        # Masks
        mask = ( ~((lam_gal > 7500)&(lam_gal<7750)) &
                 ~((lam_gal > 6810)&(lam_gal<6900)) )
        goodpixels = np.where(mask)[0]

        # -------------------------- 1-component fit -------------------------
        pp1 = self._run_ppxf_fit(
            template, self.gas_templates_1, self.gas_names_1,
            galaxy, noise, goodpixels, kbin,
            use_two_components=False
        )
        BIC1 = self._get_BIC(pp1)

        # Check if Halpha exists
        idx_Ha1 = self._get_Halpha_index(pp1.gas_names, "Halpha_(1)")
        if idx_Ha1 is None:
            # fallback: only 1-component valid
            pp2 = self._run_ppxf_fit(
                template, self.gas_templates_2, self.gas_names_2,
                galaxy, noise, goodpixels, kbin,
                use_two_components=True
            )
            return pp2, pp2.gas_flux, pp2.gas_flux_error, pp2.gas_names, pp2.chi2, [1]

        # Compute S/N(Halpha)
        f1 = pp1.gas_flux[idx_Ha1]
        e1 = pp1.gas_flux_error[idx_Ha1]
        sn_Ha = f1/e1 if e1 > 0 else 0.0

        # -------------------------- 2-component fit -------------------------
        pp2 = self._run_ppxf_fit(
            template, self.gas_templates_2, self.gas_names_2,
            galaxy, noise, goodpixels, kbin,
            use_two_components=True
        )
        BIC2 = self._get_BIC(pp2)
        dBIC = BIC1 - BIC2

        # Extract (1) and (2)
        idx_Ha1_2 = self._get_Halpha_index(pp2.gas_names, "Halpha_(1)")
        idx_Ha2_2 = self._get_Halpha_index(pp2.gas_names, "Halpha_(2)")
        if (idx_Ha1_2 is None) or (idx_Ha2_2 is None):
            return pp2, pp2.gas_flux, pp2.gas_flux_error, pp2.gas_names, pp2.chi2, [1]

        # Flux fraction test
        f1_2 = pp2.gas_flux[idx_Ha1_2]
        f2_2 = pp2.gas_flux[idx_Ha2_2]
        frac = f2_2 / (f1_2 + f2_2) if (f1_2 + f2_2) > 0 else 0.0

        # Final decision
        if (sn_Ha >= sn_min) and (dBIC > dBIC_min) and (frac > frac_min):
            return pp2, pp2.gas_flux, pp2.gas_flux_error, pp2.gas_names, pp2.chi2, [1, 2]
        else:
            return pp1, pp1.gas_flux, pp1.gas_flux_error, pp1.gas_names, pp1.chi2, [1]


    # ------------------------------------------------------------------
    # Helper to run 1- or 2-component pPXF
    # ------------------------------------------------------------------
    def _run_ppxf_fit(self, template, gas_templates, gas_names,
                      galaxy, noise, goodpixels, kbin,
                      use_two_components):

        if use_two_components:
            n_lines = len(self.gas_names_base)
            component = (
                [0] + [1]*n_lines + [2]*n_lines
            )
            moments = [-4, 2, 2]
            start = [
                [self.velbin[kbin], self.sigbin[kbin], self.h3[kbin], self.h4[kbin]],
                [self.velbin[kbin], 50.0],
                [self.velbin[kbin]+80., 180.0]
            ]
            fixed = [[1,1,1,1], [0,0], [0,0]]
        else:
            n_lines = len(self.gas_names_base)
            component = [0] + [1]*n_lines
            moments = [-4, 2]
            start = [
                [self.velbin[kbin], self.sigbin[kbin], self.h3[kbin], self.h4[kbin]],
                [self.velbin[kbin], 40.0]
            ]
            fixed = [[1,1,1,1], [0,0]]

        gas_comp_flag = np.array(component) > 0

        pp = ppxf(
            templates=np.column_stack([template, gas_templates]),
            galaxy=galaxy,
            noise=noise,
            velscale=self.s.velscale,
            start=start,
            fixed=fixed,
            goodpixels=goodpixels,
            moments=moments,
            component=component,
            gas_component=gas_comp_flag,
            gas_names=gas_names,
            lam=self.lam_gal,
            lam_temp=self.sps.lam_temp,
            degree=-1, mdegree=10,
            quiet=True, plot=False
        )

        return pp


    # ------------------------------------------------------------------
    # Run ALL spaxels in parallel
    # ------------------------------------------------------------------
    def fit_cube(self, n_jobs=-1):
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self._fit_single_spaxel)(j)
            for j in tqdm(range(len(self.s.x)), desc="Fitting gas spaxels")
        )

        self.pp_saved      = [r[0] for r in results]
        self.fluxes        = [r[1] for r in results]
        self.err_fluxes    = [r[2] for r in results]
        self.gas_names_all = [r[3] for r in results]
        self.chi_all       = [r[4] for r in results]
        self.active_comps  = [r[5] for r in results]

        return self.pp_saved


    # ------------------------------------------------------------------
    # Build utility maps
    # ------------------------------------------------------------------
    def _base_name(self, g):
        return g.split("_(")[0]

    def build_line_component_map(self, gas_names):
        d = defaultdict(list)
        for idx, g in enumerate(gas_names):
            d[self._base_name(g)].append(idx)
        return d

    def build_wavelength_map(self, gas_names):
        wmap = {}
        for g, w in zip(gas_names, self.line_wave):
            wmap[self._base_name(g)] = w
        return wmap


    # ------------------------------------------------------------------
    # Process all lines and save maps
    # ------------------------------------------------------------------
    def process_all_lines(self, header):

        gas_names_global = self.pp_saved[0].gas_names
        line_map = self.build_line_component_map(gas_names_global)
        wavelength_map = self.build_wavelength_map(gas_names_global)

        for line_name in wavelength_map.keys():
            print(f"\nProcessing line: {line_name}")
            self._create_flux_maps_for_line(
                line_name=line_name,
                wavelength=wavelength_map[line_name],
                header=header
            )

        # Weighted velocities
        if "Halpha" in line_map:
            v, s = self.weighted_kinematics("Halpha")
            self._save_fits("Halpha_vel_weighted", v, header)
            self._save_fits("Halpha_sigma_weighted", s, header)


    # ------------------------------------------------------------------
    # Create flux component maps
    # ------------------------------------------------------------------
    def _create_flux_maps_for_line(self, line_name, wavelength, header):

        naxis1 = self.s.header['NAXIS1']
        naxis2 = self.s.header['NAXIS2']
        npix = naxis1*naxis2

        flux_comp = {1:np.zeros(npix), 2:np.zeros(npix)}
        flux_err  = {1:np.zeros(npix), 2:np.zeros(npix)}
        total_flux = np.zeros(npix)
        total_err2 = np.zeros(npix)

        dlam = wavelength * self.s.velscale / c_kms

        for i in range(npix):
            pp = self.pp_saved[i]

            local_idxs = [j for j,g in enumerate(pp.gas_names)
                          if g.startswith(line_name)]
            if len(local_idxs)==0:
                continue

            for j in local_idxs:
                comp = int(pp.gas_names[j].split("(")[1].split(")")[0])

                tpl_idx = 1+j
                kin = pp.component[tpl_idx]
                if kin not in self.active_comps[i]:
                    continue

                f = pp.gas_flux[j]
                ferr = pp.gas_flux_error[j]

                if f>0:
                    f_val = f*dlam
                    f_err = ferr*np.sqrt(pp.chi2)
                else:
                    f_val = 0.0
                    f_err = 0.0

                flux_comp[comp][i] = f_val
                flux_err[comp][i]  = f_err

                total_flux[i] += f_val
                total_err2[i] += f_err**2

        for comp in [1,2]:
            self._save_fits(f"{line_name}_comp{comp}_flux",
                            flux_comp[comp].reshape(naxis1,naxis2), header)
            self._save_fits(f"{line_name}_comp{comp}_flux_err",
                            flux_err[comp].reshape(naxis1,naxis2), header)

        self._save_fits(f"{line_name}_flux",
                        total_flux.reshape(naxis1,naxis2), header)
        self._save_fits(f"{line_name}_flux_err",
                        np.sqrt(total_err2).reshape(naxis1,naxis2), header)


    # ------------------------------------------------------------------
    # Weighted mean velocity & dispersion for one line
    # ------------------------------------------------------------------
    def weighted_kinematics(self, line_name):
        naxis1 = self.s.header['NAXIS1']
        naxis2 = self.s.header['NAXIS2']
        npix = naxis1*naxis2

        vmap = np.full(npix, np.nan)
        sigmap = np.full(npix, np.nan)

        for i in range(npix):
            pp = self.pp_saved[i]

            local_idxs = [j for j,g in enumerate(pp.gas_names)
                          if g.startswith(line_name)]
            if len(local_idxs)==0:
                continue

            f_list=[]; v_list=[]; s_list=[]

            for j in local_idxs:
                tpl_idx = 1+j
                kin = pp.component[tpl_idx]
                if kin not in self.active_comps[i]:
                    continue

                f = pp.gas_flux[j]
                if f<=0:
                    continue

                v = pp.sol[kin][0]
                s = pp.sol[kin][1]
                f_list.append(f); v_list.append(v); s_list.append(s)

            if len(f_list)==0:
                continue

            f = np.array(f_list)
            v = np.array(v_list)
            s = np.array(s_list)

            vmean = np.sum(f*v)/np.sum(f)
            vmap[i] = vmean

            sig2 = np.sum(f*(s**2 + (v-vmean)**2))/np.sum(f)
            sigmap[i] = np.sqrt(sig2)

        return (vmap.reshape(naxis1,naxis2),
                sigmap.reshape(naxis1,naxis2))


    # ------------------------------------------------------------------
    # Simple FITS writer
    # ------------------------------------------------------------------
    def _save_fits(self, name, data, header):
        fits.writeto(name+".fits", data=data, header=header, overwrite=True)


