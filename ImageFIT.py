#%% IMPORTS

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import copy

import time

import numpy as np
import pandas as pd
from scipy import ndimage

import astropy.units as u
from astropy import wcs
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.visualization import make_lupton_rgb
from astropy.utils.data import download_file, clear_download_cache
from astropy.utils.data import get_pkg_data_filename

from photutils import data_properties
from photutils import make_source_mask

import matplotlib.pyplot as plt
from matplotlib import colors


from unagi import config
from unagi import hsc
from unagi import plotting
import unagi.mask as msk
from unagi.task import hsc_psf
from unagi.task import hsc_cutout

from photutils import detect_threshold
from photutils import detect_sources
from photutils import deblend_sources

from astropy.convolution import Gaussian2DKernel

import pyimfit
pyimfit.__file__
print('vl2gUfTOHhNYO4cOEZGL+4X17rDJ5WgdWDIE5cqB')
pdr2 = hsc.Hsc(dr='pdr2', rerun='pdr2_wide')

start_time = time.time()

#%% MASK FUNCTIONS

def seg_remove_cen_obj(seg):
        """Remove the central object from the segmentation."""
        seg_copy = copy.deepcopy(seg)
        seg_copy[seg == seg[int(seg.shape[0] / 2.0), int(seg.shape[1] / 2.0)]] = 0

        return seg_copy

def increase_mask_regions(mask, method='uniform', size=7, mask_threshold=0.01):
        """Increase the size of the mask regions using smoothing algorithm."""
        mask_arr = mask.astype('int16')
        mask_arr[mask_arr > 0] = 100
    
        if method == 'uniform' or method == 'box':
            mask_new = ndimage.uniform_filter(mask_arr, size=size)
        elif method == 'gaussian':
            mask_new = ndimage.gaussian_filter(
                mask_arr, sigma=size, order=0)
        else:
            raise ValueError("Wrong method. Should be uniform or gaussian.")
    
        mask_new[mask_new < mask_threshold] = 0
        mask_new[mask_new >= mask_threshold] = 1
    
        return mask_new.astype('uint8')
    
def save_to_fits(img, fits_file, wcs=None, header=None, overwrite=True):
        """Save an image to FITS file."""
        if wcs is not None:
            wcs_header = wcs.to_header()
            img_hdu = fits.PrimaryHDU(img, header=wcs_header)
        else:
            img_hdu = fits.PrimaryHDU(img)
        #if header is not None:
            #if 'SIMPLE' in header and 'BITPIX' in header:
                #img_hdu.header = header
        #else:
            #img_hdu.header.extend(header)

        if os.path.islink(fits_file):
            os.unlink(fits_file)

        img_hdu.writeto(fits_file, overwrite=overwrite)

        return

def get_mask(galaxy):

    RA, Dec, redshift = galaxyInfo[galaxy,0], galaxyInfo[galaxy,1], galaxyInfo[galaxy,2]

    coord = SkyCoord(RA, Dec, frame='icrs', unit='deg')
    
    s_phy = 200.0 * u.kpc

    filters = 'i'
    
    label_list = [r"$\rm {}$".format(label) for label in ['Image', 'Mask', 'Variance']]
    
    i_img = cutout[1].data
    i_msk = cutout[2].data
    i_var = cutout[3].data
    
    i_sig = np.sqrt(i_var)
    i_s2n = i_img / i_sig
    
    mask_galaxy = msk.Mask(i_msk, data_release='pdr2')
    
    mask_detect = mask_galaxy.extract('DETECTED')
    mask_bad = mask_galaxy.combine(['NO_DATA', 'SAT'])
    
    threshold = 2.0
    gaussian_sigma = 1.0
    npixels = 5
    nlevels = 64
    contrast = 0.0001
    
    i_thr = detect_threshold(i_img, threshold, background=None, error=i_sig, mask=mask_bad)
    kernel = Gaussian2DKernel(gaussian_sigma, x_size=5, y_size=5)
    kernel.normalize()
    
    i_seg = detect_sources(
        i_img, i_thr, npixels=npixels, connectivity=8, filter_kernel=kernel)
    
    i_seg = deblend_sources(
        i_img, i_seg, npixels=npixels, filter_kernel=kernel, 
        nlevels=nlevels, contrast=contrast, relabel=False)
    
    seg_cmap = plotting.random_cmap(ncolors=256)

    i_obj = seg_remove_cen_obj(i_seg.data)

    mask_obj = increase_mask_regions(i_obj, method='gaussian', size=3, mask_threshold=3)
    
    cen_mask = create_circular_mask(i_img, radius=100)
    mask_hsc = copy.deepcopy(mask_detect)
    mask_hsc[cen_mask] = False

    mask_hsc_new = increase_mask_regions(
        mask_hsc, method='uniform', size=11, mask_threshold=1)

    mask_final = (mask_obj | mask_hsc_new)

    _ = save_to_fits(mask_final, 'mask.fits', overwrite=True)

#%% CUTOUT FUNCTIONS

def get_cutout_image(galaxy):

    RA, Dec, redshift = galaxyInfo[galaxy,0], galaxyInfo[galaxy,1], galaxyInfo[galaxy,2]

    coord = SkyCoord(RA, Dec, frame='icrs', unit='deg')

    s_phy = 200.0 * u.kpc

    filters = 'i'

    cutout_phy = hsc_cutout(coord, cutout_size=s_phy, redshift=redshift, filters=filters, 
                        archive=pdr2, variance=True, mask=True, use_saved=False, verbose=True, save_output=False)
    
    return cutout_phy

#%% FIT FUNCTIONS

def create_circular_mask(img, center=None, radius=None):
    """Create a circular mask to apply to an image.
    
    Based on https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
    """
    h, w = img.shape
    
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def add_c0_to_sersic(sersic):
    """Make a generalized Sersic based on the best-fit model."""
    param_names = sersic.numberedParameterNames
    param_values = sersic.getRawParameters()
    
    sersic_gen = pyimfit.make_imfit_function('Sersic_GenEllipse')
    
    sersic_gen.PA.setValue(
        param_values[['PA_' in name for name in param_names]][0], [-180, 180])
    sersic_gen.ell.setValue(
        param_values[['ell_' in name for name in param_names]][0], [0.5, 0.99])
    sersic_gen.r_e.setValue(
        param_values[['r_e_' in name for name in param_names]][0], [0.5, 100.0])
    sersic_gen.n.setValue(
        param_values[['n_' in name for name in param_names]][0], [0.2, 6.0])
    sersic_gen.c0.setValue(-0.1, [-5, 5])
    
    Ie = param_values[['I_e_' in name for name in param_names]][0]
    sersic_gen.I_e.setValue(Ie, [Ie / 3.0, Ie * 100.0])
    
    return sersic_gen


def imfit_prepare_data(directory, img_file, msk_file, archive=pdr2, hsc_band='i', verbose=False):
    """Prepare imaging data for imfit fitting."""
    img_wcs = WCS(cutout[1].header)
    img, var = cutout[1].data, cutout[3].data

    # We also want to know the size of the image and the pixel coordinate for the galaxy center
    img_shape = img.shape
    cen_x, cen_y = img_shape[0] / 2., img_shape[1] / 2.

    # Also need to know the central flux level
    cen_flux = img[int(cen_x), int(cen_y)]

    # Measure background
    mask = make_source_mask(img, nsigma=1.5, npixels=4, dilate_size=15)
    bkg_avg, bkg_med, bkg_std = sigma_clipped_stats(img, sigma=2.5, mask=mask)
    
    # Define a coordinate using the center of the galaxy
    cen_coord = img_wcs.wcs_pix2world([[cen_x, cen_y]], 0)
    cen_ra, cen_dec = cen_coord[0][0], cen_coord[0][1]
    coord = SkyCoord(cen_ra, cen_dec, frame='icrs', unit='deg')
    
    # Initial estimates of the galaxy size, shape, and orientation
    cat = data_properties(
        img[int(cen_x - 150): int(cen_x + 150), int(cen_y - 150): int(cen_y + 150)], 
        mask=mask.astype(bool)[
            int(cen_x - 150): int(cen_x + 150), int(cen_y - 150): int(cen_y + 150)], 
        background=bkg_med)
    columns = ['id', 'semimajor_axis_sigma', 'semiminor_axis_sigma', 'orientation']
    tbl = cat.to_table(columns=columns)

    if verbose:
        txt_file = open(directory,"w")
        txt_list = ["# RA, Dec: {:f}, {:f}\n".format(cen_ra, cen_dec)]
        txt_list.append("# Mean sky background: {:f}\n".format(bkg_med))
        txt_list.append("# Uncertainty of sky background: {:f}\n".format(bkg_std))
        txt_list.append("# Major axis length: {:f} pixel\n".format(
            tbl['semimajor_axis_sigma'][0].value))
        txt_list.append("# Minor axis length: {:f} pixel\n".format(
            tbl['semiminor_axis_sigma'][0].value))
        txt_list.append("# Position angle: {:f}\n".format(
            tbl['semiminor_axis_sigma'][0].value + 90.0))
        txt_file.writelines(txt_list)
        txt_file = open(directory)
        
        #content = txt_file.read()
        txt_file.close()
        #print(content)

    # This command will download PSF image from HSC archive. 
    # It will take a few seconds, so it is better to download all the necessary ones at once.
    psf_model = hsc_psf(coord, filters=hsc_band, archive=archive, save_output=False)
    psf = psf_model[0].data
    
    # Put all useful information in a dict
    return {'img': img, 'msk': mask, 'var': var, 'psf': psf,
            'cen_x': cen_x, 'cen_y': cen_y, 'cen_flux': cen_flux, 
            'gal_a': tbl['semimajor_axis_sigma'][0].value,
            'gal_b': tbl['semiminor_axis_sigma'][0].value,
            'gal_pa': tbl['semiminor_axis_sigma'][0].value + 90.,
            'bkg_avg': bkg_avg, 'bkg_std': bkg_std, 'coord': coord}


def update_galaxy_geometry(galaxy, model, model_type='Sersic'):
    """Update the size and shape of galaxy based on single-comp model."""
    galaxy_new = copy.deepcopy(galaxy)
    param_names = model.numberedParameterNames
    param_values = model.getRawParameters()
    
    pa = param_values[['PA_' in name for name in param_names]][0]
    i0 = param_values[['I_e_' in name for name in param_names]][0] * 2.0
    ell = param_values[['ell_' in name for name in param_names]][0]
    rad = param_values[['r_e_' in name for name in param_names]][0]
        
    galaxy_new['gal_a'] = rad
    galaxy_new['gal_b'] = rad * ell
    galaxy_new['gal_pa'] = pa
    if i0 <= galaxy['cen_flux'] * 5.0:
        galaxy_new['cen_flux'] = i0
    
    return galaxy_new


def imfit_fit_sersic(iteration, Index, directory, galaxy, model_type='Sersic', solver='LM', model_type_2=None,
                     visual=True, model_ini=None, model_ini_2=None, update_sersic=None):
    
    model_name = str(model_type)
    
    """Fit an Imfit model."""
    if solver not in ['LM', 'NM', 'DE']:
        raise ValueError("# Wrong solver type: [LM|NM|DE]")

    cen_x, cen_y = galaxy['cen_x'], galaxy['cen_y']
    cen_flux = galaxy['cen_flux']
    gal_a, gal_b = galaxy['gal_a'], galaxy['gal_b']
    gal_pa = galaxy['gal_pa']
    # The ellipticity here is more appropriate for bulge, not the disk
    gal_e = 1.0 - gal_b / gal_a
    
    # Define the limits on the central-coordinate X0 and Y0 as +/-10 pixels relative to initial values
    galaxy_desc = pyimfit.SimpleModelDescription()
    galaxy_desc.x0.setValue(cen_x, [cen_x - 10, cen_x + 10])
    galaxy_desc.y0.setValue(cen_y, [cen_y - 10, cen_y + 10])

    if model_type == 'Exponential':
        comp1 = pyimfit.make_imfit_function('Exponential')
        comp1.PA.setValue(gal_pa, [-180, 180])
        if gal_e <= 0.6:
            comp1.ell.setValue(0.6, [0.05, 1.0])
        else:
            comp1.ell.setValue(gal_e, [0.05, 1.0])
        comp1.I_0.setValue(cen_flux, [cen_flux / 2.0, cen_flux * 200.0])
        comp1.h.setValue(gal_a / np.e, [0.1, 100])
    elif model_type == 'Sersic':
        comp1 = pyimfit.make_imfit_function('Sersic')
        comp1.PA.setValue(gal_pa, [-180, 180])
        if gal_e <= 0.6:
            comp1.ell.setValue(0.6, [0.05, 1.0])
        else:
            comp1.ell.setValue(gal_e, [0.05, 1.0])
        comp1.I_e.setValue(cen_flux * 2.0, [cen_flux / 10.0, cen_flux * 200.0])
        comp1.r_e.setValue(gal_a / 2.0, [0.5, 100])
        comp1.n.setValue(1.2, [0.2, 6.0])
    else:
        raise ValueError("# Wrong model type!")
        
    galaxy_desc.addFunction(comp1)
    
    # Add optional second model
    if model_type_2 is not None:
        if model_type_2 == 'Sersic':
            model_name = str(model_type)+" + "+str(model_type_2)
            comp2 = pyimfit.make_imfit_function('Sersic')
            comp2.PA.setValue(gal_pa, [-180, 180])
            if gal_e >= 0.3:
                comp2.ell.setValue(0.3, [0.02, 0.99])
            else:
                comp2.ell.setValue(gal_e, [0.02, 0.99])
            comp2.I_e.setValue(cen_flux * 2., [cen_flux / 20.0, cen_flux * 100.0])
            comp2.r_e.setValue(4.0, [0.2, 100])
            comp2.n.setValue(3.0, [0.2, 6.0])
        else:
            raise ValueError("# Wrong model type!")
            
        galaxy_desc.addFunction(comp2)

    # Perform the fitting
    galaxy_model = pyimfit.Imfit(galaxy_desc, galaxy['psf'])
    _ = galaxy_model.fit(
        galaxy['img'], mask=galaxy['msk'], error=galaxy['var'], error_type='variance', 
        solver=solver, verbose=1)

    # Check the result
    if galaxy_model.fitConverged:
        txt_file = open(directory,"a")
        txt_list = ["\n"]
        txt_list.append(str(model_name)+":\n")
        txt_list.append("# Chi2 statistics: {:f}\n".format(galaxy_model.fitStatistic))
        arrNew[0,Index] = galaxy_model.fitStatistic
        txt_list.append("# Reduced Chi2 statistics: {:f}\n".format(galaxy_model.reducedFitStatistic))
        arrNew[0,Index+1] = galaxy_model.reducedFitStatistic
        txt_list.append("# AIC statistics: {:f}\n".format(galaxy_model.AIC))
        arrNew[0,Index+2] = galaxy_model.AIC
        txt_list.append("# BIC statistics: {:f}\n\n".format(galaxy_model.BIC))
        arrNew[0,Index+3] = galaxy_model.BIC
        txt_list.append("# Best fit parameter values:\n")
        
        if solver == 'LM':
            for name, val, err in zip(
                galaxy_model.numberedParameterNames, galaxy_model.getRawParameters(), 
                galaxy_model.getParameterErrors()): 
                txt_list.append("   {:7s}: {:f}+/-{:f}\n".format(name, val, err))
        else:
            for name, val in zip(
                galaxy_model.numberedParameterNames, galaxy_model.getRawParameters()): 
                txt_list.append("   {:7s}: {:f}\n".format(name, val))

        txt_file.writelines(txt_list)
        txt_file = open(directory)
        
        txt_file.close()

        if visual:
            # Residual map
            galaxy_mod = galaxy_model.getModelImage()
            galaxy_res = galaxy['img'] - galaxy_mod
            galaxy_chi = galaxy_res * np.sqrt(galaxy['var'])

            # Clear out the inner region
            cen_mask = create_circular_mask(galaxy['img'], radius=120)
            galaxy_residual = galaxy_chi[cen_mask & (galaxy['msk'] == 0)]
            
            # Visualize the residual map
            fig = plt.figure(figsize=(9, 4.5))
            fig.subplots_adjust(left=0.0, right=0.965, bottom=0.15, top=0.99, wspace=0.0, hspace=0.0)

            # Highlight the residual pattern around the galaxy
            ax1 = fig.add_subplot(121)
            ax1.grid(False)

            # We use a different colormap to highlight features on the residual map. 
            # We can use blue for negative values and red for positive values
            ax1 = plotting.display_single(
                galaxy_chi[int(cen_x - 150):int(cen_x + 150), int(cen_y - 150):int(cen_y + 150)], 
                cmap='RdBu_r', stretch='arcsinh', zmin=-0.15, zmax=0.15, ax=ax1,
                scale_bar_color='k', scale_bar_y_offset=0.3)
            ax1.imshow(
                galaxy['msk'].astype('float')[
                    int(cen_x - 150):int(cen_x + 150), int(cen_y - 150):int(cen_y + 150)], 
                origin='lower', interpolation='none', cmap='Greys', alpha=0.1)

            # Show the distribution of residual levels
            ax2 = fig.add_subplot(122)
            ax2.axvline(0.0, linestyle='--', color='k', alpha=0.9)
            _ = ax2.hist(galaxy_residual, bins=100, log=True, histtype='stepfilled',
                         density=True, alpha=0.5, edgecolor='k')
            ax2.set_yticklabels([])
            ax2.set_xlabel(r'$\rm (Data - Model) / \sigma$', fontsize=25)
            ax2.text(0.95, 0.9, r'$\rm {:s}$'.format(model_type.replace('_', '')), fontsize=20, 
                     transform=ax2.transAxes, ha='right')
            if model_type_2 is not None:
                ax2.text(0.95, 0.83, r'$+ \ \rm {:s}$'.format(model_type_2.replace('_', '')), 
                         fontsize=20, transform=ax2.transAxes, ha='right')
                
            plt.close()
                
            fig.savefig('/Users/colemeyer/Documents/Isophotal Twist/Pipe Out/Images/'+galaxyName+str(iteration))
    else:
        print("# Model is not converged! Please try again!")
    
    return galaxy_model

#%% LOOP

n = 4
stats = np.zeros((n,13))

galaxyInfo = np.loadtxt(open("input.csv", encoding='utf-8-sig'), delimiter=",")


dir_in = '/Users/colemeyer/Documents/Isophotal Twist/Pipe In/'
dir_out = '/Users/colemeyer/Documents/Isophotal Twist/Pipe Out/'

for galaxy in range(1,n+1):
    arrNew = np.zeros((1,13))
    arrNew[0,0] = galaxy
    
    galaxyName = "Galaxy" + str(galaxy)
    textDirectory = dir_out+'Text Files/'+galaxyName+".txt"
    
    cutout = get_cutout_image(galaxy-1)
    get_mask(galaxy-1)
    mask = fits.open('/Users/colemeyer/Documents/Isophotal Twist/Pipe In/mask.fits')
    
    galaxy = imfit_prepare_data(
        textDirectory, cutout, mask, verbose=True)
    
    sersicfit_model_1 = imfit_fit_sersic("-1", 1, textDirectory, galaxy)
    
    sersicfit_new = update_galaxy_geometry(galaxy, sersicfit_model_1)
    
    sersicfit_model_2 = imfit_fit_sersic("-2", 5, textDirectory, sersicfit_new, model_type_2='Sersic')
    
    sersicfit_model_3 = imfit_fit_sersic("-3", 9, textDirectory, sersicfit_new, model_type='Exponential', model_type_2='Sersic')
    
    stats = np.vstack((stats, arrNew))
    
    if os.path.exists("mask.fits"):
        os.remove("mask.fits")
    else:
        print("The file does not exist")
    
for i in range (n):
    stats = np.delete(stats, 0, 0)
DF = pd.DataFrame(stats)
DF.to_csv("/Users/colemeyer/Documents/Isophotal Twist/Pipe Out/data.csv",header=
          ['Galaxy #:','Chi2:','R-Chi2:','AIC:','BIC:','Chi2:','R-Chi2:','AIC:',
           'BIC:','Chi2:','R-Chi2:','AIC:','BIC:',],index=False)

end_time = time.time()
elapsed_time = end_time - start_time

print("\nTime Elapsed: "'%.2f' % elapsed_time)

