from mpdaf.obj import Cube, Image
import matplotlib.pyplot as plt
from astropy.io import fits

class AstrometryCorrector:
    """
    Apply WCS astrometric corrections to a MUSE (or general FITS) cube
    by matching it to a reference image.
    """

    def __init__(self, cube_filename, reference_image_file):
        """
        cube_filename: path to the FITS cube
        reference_image_file: path to the reference image FITS file
        """
        self.cube_filename = cube_filename
        self.reference_image_file = reference_image_file

        # load objects
        self.ref_img = Image(reference_image_file)
        self.fullcube = Cube(cube_filename)

    # -----------------------------------------------------------------

    def show_reference(self):
        """Plot the reference image."""
        plt.figure(figsize=(6, 6))
        self.ref_img.plot(use_wcs=True, cmap="gray_r", scale="arcsinh")
        plt.title("Reference Image")
        plt.show()

    # -----------------------------------------------------------------

    def show_white_image(self):
        """Plot the collapsed (white-light) image of the cube."""
        ima = self.fullcube.sum(axis=0)

        plt.figure(figsize=(6,6))
        ima.plot(use_wcs=True, cmap="gray_r", scale="arcsinh")
        plt.title("Cube White-Light Image")
        plt.show()

    # -----------------------------------------------------------------

    def compute_offset(self):
        """Estimate WCS offset between cube and reference."""
        ima = self.fullcube.sum(axis=0)
        offset = ima.estimate_coordinate_offset(self.ref_img, nsigma=1)
        print("Estimated offset (dy, dx):", offset)
        return offset

    # -----------------------------------------------------------------

    def apply_correction(self):
        """
        Apply the astrometric correction to CRPIX values
        and write a new WCS-aligned cube.
        """
        ima = self.fullcube.sum(axis=0)
        offset = self.compute_offset()

        # new CRPIX values
        ima_crpix1 = ima.wcs.get_crpix1() + offset[1]
        ima_crpix2 = ima.wcs.get_crpix2() + offset[0]

        print("New CRPIX1, CRPIX2:", ima_crpix1, ima_crpix2)

        # OPEN FITS FILE DIRECTLY
        hdul = fits.open(self.cube_filename)
        print("\nExtensions found:")
        hdul.info()

        for i, hdu in enumerate(hdul):

            extname = hdu.header.get("EXTNAME", "").upper()

            # Update DATA (HDU 1 in MUSE)
            if extname == "DATA":
                print(f"  ✓ Updating DATA (HDU {i})")
                hdu.header["CRPIX1"] = ima_crpix1
                hdu.header["CRPIX2"] = ima_crpix2
                updated_data = True

            # Update STAT or VAR depending on naming
            elif extname in ["STAT", "VAR", "VARIANCE"]:
                print(f"  ✓ Updating {extname} (HDU {i})")
                hdu.header["CRPIX1"] = ima_crpix1
                hdu.header["CRPIX2"] = ima_crpix2
                updated_stat = True

            else:
                print(f"  – Skipping HDU {i} ({extname or 'PRIMARY'})")

        # ------------------------------------------------------------------
        # SAVE NEW FILE
        # ------------------------------------------------------------------
        new_filename = self.cube_filename.replace(".fits", "_wcs_aligned.fits")
        hdul.writeto(new_filename, overwrite=True)
        hdul.close()

        print("\nSaved:", new_filename)
        return new_filename


    #def close(self):
    #    """Close cube and release memory."""
     #   self.fullcube.close()


'''
def astrometry_correction(cube_filename,reference_image_file):



    ref_img = Image(reference_image_file)
    plt.figure(figsize=(6,6))
    ref_img.plot(use_wcs=True, cmap="gray_r", scale='arcsinh')
    plt.show()


    fullcube = Cube(cube_filename)
    ima = fullcube.sum(axis=0)

    plt.figure(figsize=(6,6))
    ima.plot(use_wcs=True, cmap="gray_r",scale='arcsinh')
    ima.write("WhiteImage.fits")
    #plt.show()

    # Estimate offset
    offset = ima.estimate_coordinate_offset(ref_img, nsigma=1)
    print(offset)

    ima_crpix1 = ima.wcs.get_crpix1() + offset[1]
    ima_crpix2 = ima.wcs.get_crpix2() + offset[0]

    print(ima_crpix1, ima_crpix2)

    header1 = fullcube[1].header
    header2 = fullcube[2].header
    #header3 = fullcube[3].header

    print(" data:", header1["CRPIX1"], header1["CRPIX2"], "\n",
        "stat:",header2["CRPIX1"], header2["CRPIX2"], "\n")

    header1["CRPIX1"] = ima_crpix1
    header1["CRPIX2"] = ima_crpix2
    header2["CRPIX1"] = ima_crpix1
    header2["CRPIX2"] = ima_crpix2

    cube_filename_new = cube_filename.replace(".fits", "_wcs_aligned.fits")

    fullcube.writeto(cube_filename_new, overwrite=True)
    fullcube.close()
'''