import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import copy

import time

import numpy as np
from scipy import ndimage

import astropy.units as u
from astropy import wcs
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.visualization import make_lupton_rgb
from astropy.utils.data import download_file, clear_download_cache

from photutils import data_properties
from photutils import make_source_mask

import matplotlib.pyplot as plt
from matplotlib import colors


from unagi import config
from unagi import hsc
from unagi import plotting
from unagi.task import hsc_psf

import pyimfit
pyimfit.__file__
print('vl2gUfTOHhNYO4cOEZGL+4X17rDJ5WgdWDIE5cqB')
pdr2 = hsc.Hsc(dr='pdr2', rerun='pdr2_wide')

start_time = time.time()

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

#%% Automation


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
    cutout = fits.open(img_file)
    img_wcs = WCS(cutout[1].header)
    img, var = cutout[1].data, cutout[3].data
    msk = fits.open(msk_file)[0].data

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
        mask=msk.astype(bool)[
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
    return {'img': img, 'msk': msk, 'var': var, 'psf': psf,
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


def imfit_fit_sersic(iteration, directory, galaxy, model_type='Sersic', solver='LM', model_type_2=None,
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
        txt_list.append("# Reduced Chi2 statistics: {:f}\n".format(galaxy_model.reducedFitStatistic))
        txt_list.append("# AIC statistics: {:f}\n".format(galaxy_model.AIC))
        txt_list.append("# BIC statistics: {:f}\n\n".format(galaxy_model.BIC))
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
        
        #content = txt_file.read()
        txt_file.close()
        #print(content)

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

dir_in = '/Users/colemeyer/Documents/Isophotal Twist/Pipe In/'
dir_out = '/Users/colemeyer/Documents/Isophotal Twist/Pipe Out/'

for galaxy in range(1,5):
    galaxyName = "Galaxy" + str(galaxy)
    textDirectory = dir_out+'Text Files/'+galaxyName+".txt"
    
    galaxy = imfit_prepare_data(
        textDirectory, os.path.join(dir_in, 'A'+galaxyName+'.fits'), os.path.join(dir_in, 'C'+galaxyName+'.fits'), verbose=True)
    
    sersicfit_model_1 = imfit_fit_sersic("-1", textDirectory, galaxy)
    
    sersicfit_new = update_galaxy_geometry(galaxy, sersicfit_model_1)
    
    sersicfit_model_2 = imfit_fit_sersic("-2", textDirectory, sersicfit_new, model_type_2='Sersic')
    
    sersicfit_model_3 = imfit_fit_sersic("-3", textDirectory, sersicfit_new, model_type='Exponential', model_type_2='Sersic')
    
end_time = time.time()
elapsed_time = end_time - start_time

print("\nTime Elapsed: "'%.2f' % elapsed_time)