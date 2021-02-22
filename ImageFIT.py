import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import copy

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
'''
#%% STARTING PARAMETERS

file = 'Galaxy1.fits'
ra_obj, dec_obj = 139.0438589, 3.362059



cutout_edge = fits.open('A'+file)

# Science image
i_img = cutout_edge[1].data

# Variance map; both are 2-D numpy array
i_var = cutout_edge[3].data

_ = plotting.display_single(i_img)


# We also want to know the size of the image and the pixel coordinate for the galaxy center
img_shape = i_img.shape

cen_x, cen_y = img_shape[0] / 2., img_shape[1] / 2.

print(img_shape, cen_x, cen_y)

# Also need to know the central flux level
cen_flux = i_img[int(cen_x), int(cen_y)]


# We also want to know the average sky background value and its scatter 
mask = make_source_mask(i_img, nsigma=1.5, npixels=4, dilate_size=15)
bkg_avg, bkg_med, bkg_std = sigma_clipped_stats(i_img, sigma=2.5, mask=mask)

print("# Mean sky background: {:f}".format(bkg_med))
print("# Uncertainty of sky background: {:f}".format(bkg_std))

i_msk = fits.open('C'+file)[0].data


# Define a coordinate using the center of the galaxy
coord = SkyCoord(ra_obj, dec_obj, frame='icrs', unit='deg')

# This command will download PSF image from HSC archive. 
# It will take a few seconds, so it is better to download all the necessary ones at once.
psf_model = hsc_psf(coord, filters='i', archive=pdr2, prefix=file)

i_psf = psf_model[0].data

# PSF looks boring. It is basically a Gaussian function with some small deviations. 
# Basically, it is what a star looks like on HSC image
_ = plotting.display_single(i_psf, contrast=0.1, scale_bar_length=1.0)


# Let call this model `galaxy`
galaxy_desc = pyimfit.SimpleModelDescription()

# Define the limits on the central-coordinate X0 and Y0 as +/-10 pixels relative to initial values
galaxy_desc.x0.setValue(cen_x, [cen_x - 10, cen_x + 10])
galaxy_desc.y0.setValue(cen_y, [cen_y - 10, cen_y + 10])

# Creates a Sersic component
sersic = pyimfit.make_imfit_function('Sersic')
sersic.PA.setValue(125, [0, 180])
# Ellipticity should be between 0 and 1.0
sersic.ell.setValue(0.5, [0, 1.0])
sersic.I_e.setValue(cen_flux / 2.0, [cen_flux / 10.0, cen_flux * 20.0])
sersic.r_e.setValue(15.0, [0.5, 50])
sersic.n.setValue(3, [0.5, 4.0])

galaxy_desc.addFunction(sersic)
print(img_shape)
# We want to pass the PSF image to the model object now
galaxy = pyimfit.Imfit(galaxy_desc, i_psf)

# And we can see what's our initial guess look like
model_ini = galaxy.getModelImage(shape=img_shape)

# Not crazy compared to the real galaxy
_ = plotting.display_single(model_ini, scale='linear', contrast=1.0)



# Just for demonstration, we can also get a version without PSF convolution
galaxy_nopsf = pyimfit.Imfit(galaxy_desc) 

# And we can generate a model without PSF convolution and check the difference
model_nopsf = galaxy_nopsf.getModelImage(shape=img_shape)

# Not crazy compared to the real galaxy
_ = plotting.display_single(model_ini - model_nopsf, contrast=0.5)

# Basically, PSF "moves" some flux from the high-intensity region to the fainter part of the galaxy.

galaxy.loadData(i_img, mask=i_msk, error=i_var, error_type="variance")

solver = 'LM'

galaxy_result = galaxy.doFit(solver=solver)

if galaxy_result.fitConverged:
    # If the fitting result converged (meaning the algorithm thinks it finds the best solution)
    # Show some fitting statistics
    print("# Chi2 satistics: {:f}".format(galaxy_result.fitStat))
    # For a perfect model reduced chi2 value should be very close to 1.0
    print("# Reduced Chi2 satistics: {:f}".format(galaxy_result.fitStatReduced))
    # AIC and BIC
    print("# AIC statistics: {:f}".format(galaxy_result.aic))
    print("# BIC statistics: {:f}".format(galaxy_result.bic))
    print("\n # Best fit parameter values:")
    if solver == 'LM':
        for name, val, err in zip(
            galaxy.numberedParameterNames, galaxy_result.params, galaxy_result.paramErrs): 
            print("   {:7s}: {:f}+/-{:f}".format(name, val, err))
    else:
        for name, val in zip(galaxy.numberedParameterNames, galaxy_result.params): 
            print("   {:7s}: {:f}".format(name, val))
else:
    print("# Model is not converged! Please try again!")

# Now let's see the model image
galaxy_mod = galaxy.getModelImage()

# Model image contains no noise, and the "background" value is determined by the numerical accuracy of some C++ code
# used in `imfit`. Something like 1E-10..this will cause problem when we try to visualize the model image directly
# We can create a fake "sky" image to add to it

i_sky = np.random.normal(loc=bkg_avg, scale=bkg_std, size=img_shape)

_ = plotting.display_single(galaxy_mod + i_sky, contrast=0.15)

# The model and model parameters do not look crazy


galaxy_res = i_img - galaxy_mod

# This is the relative model error information.
# It is (data - model) / error
galaxy_chi = galaxy_res * np.sqrt(i_var)


cen_mask = create_circular_mask(i_img, radius=120)

galaxy_residual = galaxy_chi[cen_mask & (i_msk == 0)]

fig = plt.figure(figsize=(9, 4.5))
fig.subplots_adjust(left=0.0, right=1.0, bottom=0.1, top=0.99, wspace=0.0, hspace=0.0)

# Highlight the residual pattern around the galaxy
ax1 = fig.add_subplot(121)
ax1.grid(False)

# We use a different colormap to highlight features on the residual map. 
# We can use blue for negative values and red for positive values
ax1 = plotting.display_single(
    galaxy_chi[int(cen_x - 120):int(cen_x + 120), int(cen_y - 120):int(cen_y + 120)], 
    cmap='RdBu_r', stretch='arcsinh', zmin=-0.15, zmax=0.15, ax=ax1,
    scale_bar_color='k', scale_bar_y_offset=0.3)
ax1.imshow(
    i_msk.astype('float')[int(cen_x - 120):int(cen_x + 120), int(cen_y - 120):int(cen_y + 120)], 
    origin='lower', interpolation='none', cmap='Greys', alpha=0.1)

# Show the distribution of residual levels
ax2 = fig.add_subplot(122)
ax2.axvline(0.0, linestyle='--', color='k', alpha=0.9)
_ = ax2.hist(galaxy_residual, bins=100, log=True, histtype='stepfilled',
             density=True, alpha=0.5, edgecolor='k')
ax2.set_yticklabels([])

ax2.set_xlabel(r'$\rm (Data - Model) / \sigma$', fontsize=25)


#%% Multi-Component Fitting
# Let call this model `buldgedisk`
galaxy2_desc = pyimfit.SimpleModelDescription()

# Define the limits on the central-coordinate X0 and Y0 as +/-10 pixels relative to initial values
galaxy2_desc.x0.setValue(cen_x, [cen_x - 10, cen_x + 10])
galaxy2_desc.y0.setValue(cen_y, [cen_y - 10, cen_y + 10])

# Creates a Sersic component
sersic2 = pyimfit.make_imfit_function('Sersic')
sersic2.PA.setValue(125, [0, 180])
# Ellipticity should be between 0 and 1.0
sersic2.ell.setValue(0.5, [0, 1.0])
sersic2.I_e.setValue(cen_flux / 2.0, [cen_flux / 10.0, cen_flux * 20.0])
sersic2.r_e.setValue(15.0, [0.5, 50])
sersic2.n.setValue(3, [0.5, 4.0])

# Creates a Sersic component
sersic3 = pyimfit.make_imfit_function('Sersic')
sersic3.PA.setValue(125, [0, 180])
# Ellipticity should be between 0 and 1.0
sersic3.ell.setValue(0.5, [0, 1.0])
sersic3.I_e.setValue(cen_flux / 2.0, [cen_flux / 10.0, cen_flux * 20.0])
sersic3.r_e.setValue(15.0, [0.5, 50])
sersic3.n.setValue(3, [0.5, 4.0])

galaxy2_desc.addFunction(sersic2)
galaxy2_desc.addFunction(sersic3)

# We want to pass the PSF image to the model object now
galaxy2 = pyimfit.Imfit(galaxy2_desc, i_psf)

# And we can see what's our initial guess look like
model_ini = galaxy2.getModelImage(shape=img_shape)

# Not crazy compared to the real galaxy
_ = plotting.display_single(model_ini, scale='linear', contrast=1.0)


_ = galaxy2.fit(i_img, mask=i_msk, error=i_var, error_type='variance', solver='LM', verbose=1)

if galaxy.fitConverged:
    # If the fitting result converged (meaning the algorithm thinks it finds the best solution)
    # Show some fitting statistics
    print("# Chi2 satistics: {:f}".format(galaxy2.fitStatistic))
    # For a perfect model reduced chi2 value should be very close to 1.0
    print("# Reduced Chi2 satistics: {:f}".format(galaxy2.reducedFitStatistic))
    # AIC and BIC
    print("# AIC statistics: {:f}".format(galaxy2.AIC))
    print("# BIC statistics: {:f}".format(galaxy2.BIC))
    print("\n # Best fit parameter values:")
    if solver == 'LM':
        for name, val, err in zip(
            galaxy2.numberedParameterNames, galaxy2.getRawParameters(), 
            galaxy2.getParameterErrors()): 
            print("   {:7s}: {:f}+/-{:f}".format(name, val, err))
    else:
        for name, val in zip(galaxy2.numberedParameterNames, galaxy2.getRawParameters()): 
            print("   {:7s}: {:f}".format(name, val))
else:
    print("# Model is not converged! Please try again!")
    
# Model image
galaxy2_mod = galaxy2.getModelImage()

# Now let's see the residual patterns
galaxy2_res = i_img - galaxy2_mod

# This is the relative model error information.
# It is (data - model) / error
galaxy2_chi = galaxy2_res * np.sqrt(i_var)

# Clear out the inner region
cen_mask = create_circular_mask(i_img, radius=120)

galaxy_residual = galaxy2_chi[cen_mask & (i_msk == 0)]

fig = plt.figure(figsize=(9, 4.5))
fig.subplots_adjust(left=0.0, right=1.0, bottom=0.1, top=0.99, wspace=0.0, hspace=0.0)

# Highlight the residual pattern around the galaxy
ax1 = fig.add_subplot(121)
ax1.grid(False)

# We use a different colormap to highlight features on the residual map. 
# We can use blue for negative values and red for positive values
ax1 = plotting.display_single(
    galaxy2_chi[int(cen_x - 120):int(cen_x + 120), int(cen_y - 120):int(cen_y + 120)], 
    cmap='RdBu_r', stretch='arcsinh', zmin=-0.15, zmax=0.15, ax=ax1,
    scale_bar_color='k', scale_bar_y_offset=0.3)
ax1.imshow(
    i_msk.astype('float')[int(cen_x - 120):int(cen_x + 120), int(cen_y - 120):int(cen_y + 120)], 
    origin='lower', interpolation='none', cmap='Greys', alpha=0.1)

# Show the distribution of residual levels
ax2 = fig.add_subplot(122)
ax2.axvline(0.0, linestyle='--', color='k', alpha=0.9)
_ = ax2.hist(galaxy_residual, bins=100, log=True, histtype='stepfilled',
             density=True, alpha=0.5, edgecolor='k')
ax2.set_yticklabels([])

ax2.set_xlabel(r'$\rm (Data - Model) / \sigma$', fontsize=25)
'''

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


def imfit_prepare_data(img_file, msk_file, archive=pdr2, hsc_band='i', verbose=False):
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
        print("# RA, Dec: {:f}, {:f}".format(cen_ra, cen_dec))
        print("# Mean sky background: {:f}".format(bkg_med))
        print("# Uncertainty of sky background: {:f}".format(bkg_std))
        print("# Major axis length: {:f} pixel".format(
            tbl['semimajor_axis_sigma'][0].value))
        print("# Minor axis length: {:f} pixel".format(
            tbl['semiminor_axis_sigma'][0].value))
        print("# Position angle: {:f}".format(
            tbl['semiminor_axis_sigma'][0].value + 90.0))

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
    
    if model_type == 'Sersic' or model_type == 'Sersic_GenEllipse':
        pa = param_values[['PA_' in name for name in param_names]][0]
        i0 = param_values[['I_e_' in name for name in param_names]][0] * 2.0
        ell = param_values[['ell_' in name for name in param_names]][0]
        rad = param_values[['r_e_' in name for name in param_names]][0]
    elif model_type == 'BrokenExponential':
        pa = param_values[['PA_' in name for name in param_names]][0]
        i0 = param_values[['I_0_' in name for name in param_names]][0] / 4.0
        ell = param_values[['ell_' in name for name in param_names]][0]
        rad = param_values[['h2_' in name for name in param_names]][0] * 1.5
    elif model_type == 'Exponential':
        pa = param_values[['PA_' in name for name in param_names]][0]
        i0 = param_values[['I_0_' in name for name in param_names]][0] / 4.0
        ell = param_values[['ell_' in name for name in param_names]][0]
        rad = param_values[['h1_' in name for name in param_names]][0] * 1.5
    else:
        raise ValueError("# Wrong model type!")
        
    galaxy_new['gal_a'] = rad
    galaxy_new['gal_b'] = rad * ell
    galaxy_new['gal_pa'] = pa
    if i0 <= galaxy['cen_flux'] * 5.0:
        galaxy_new['cen_flux'] = i0
    
    return galaxy_new


def imfit_fit_edgeon(galaxy, model_type='BrokenExponential', solver='LM', model_type_2=None,
                     visual=True, model_ini=None, model_ini_2=None, update_sersic=None):
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

    if model_type == 'EdgeOnDisk':
        # Creates an Edge-on Exponential component
        comp1 = pyimfit.make_imfit_function('EdgeOnDisk')
        comp1.PA.setValue(gal_pa, [-180, 180])
        comp1.L_0.setValue(cen_flux * 2.0, [cen_flux / 2.0, cen_flux * 200.0])
        comp1.h.setValue(gal_a / np.e, [0.2, 100])
        comp1.z_0.setValue(5., [0.2, 50])
        comp1.n.setValue(10, [0.5, 50000])
    elif model_type == 'BrokenExponential':
        comp1 = pyimfit.make_imfit_function('BrokenExponential')
        comp1.PA.setValue(gal_pa, [-180, 180])
        if gal_e <= 0.6:
            comp1.ell.setValue(0.6, [0.05, 1.0])
        else:
            comp1.ell.setValue(gal_e, [0.05, 1.0])
        comp1.I_0.setValue(cen_flux * 2.0, [cen_flux / 2.0, cen_flux * 200.0])
        comp1.h1.setValue(gal_a / 8., [0.2, 100])
        comp1.h2.setValue(gal_a / np.e, [0.2, 100])
        comp1.r_break.setValue(gal_a / 4., [0.02, 100])
        comp1.alpha.setValue(1.0, [0.05, 1000.0])
    elif model_type == 'BrokenExponential2D':
        comp1 = pyimfit.make_imfit_function('BrokenExponential2D')
        comp1.PA.setValue(gal_pa, [-180, 180])
        comp1.I_0.setValue(cen_flux * 2.0, [cen_flux / 2.0, cen_flux * 200.0])
        comp1.h1.setValue(gal_a / 8.0, [0.1, 100])
        comp1.h2.setValue(gal_a / np.e, [0.2, 100])
        comp1.r_break.setValue(gal_a / 4., [0.02, 100])
        comp1.alpha.setValue(1.0, [0.05, 1000.0])
        comp1.h_z.setValue(4., [0.1, 50])
    elif model_type == 'ExponentialDisk3D':
        comp1 = pyimfit.make_imfit_function('ExponentialDisk3D')
        comp1.PA.setValue(gal_pa, [-180, 180])
        comp1.inc.setValue(85.0, [0, 180.0])
        comp1.J_0.setValue(cen_flux * 2.0, [cen_flux / 2.0, cen_flux * 200.0])
        comp1.h.setValue(gal_a / np.e, [0.1, 50])
        comp1.n.setValue(10, [0.5, 50000])
        comp1.z_0.setValue(5., [0.2, 50])
    elif model_type == 'Exponential':
        comp1 = pyimfit.make_imfit_function('Exponential')
        comp1.PA.setValue(gal_pa, [-180, 180])
        if gal_e <= 0.6:
            comp1.ell.setValue(0.6, [0.05, 1.0])
        else:
            comp1.ell.setValue(gal_e, [0.05, 1.0])
        comp1.I_0.setValue(cen_flux, [cen_flux / 2.0, cen_flux * 200.0])
        comp1.h.setValue(gal_a / np.e, [0.1, 100])
    elif model_type == 'Exponential_GenEllipse':
        comp1 = pyimfit.make_imfit_function('Exponential_GenEllipse')
        comp1.PA.setValue(gal_pa, [-180, 180])
        if gal_e <= 0.6:
            comp1.ell.setValue(0.6, [0.05, 1.0])
        else:
            comp1.ell.setValue(gal_e, [0.05, 1.0])
        comp1.I_0.setValue(cen_flux, [cen_flux / 2.0, cen_flux * 200.0])
        comp1.h.setValue(gal_a / np.e, [0.1, 100])
        comp1.c0.setValue(0, [-4, 4])
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
    elif model_type == 'Sersic_GenEllipse':
        if update_sersic is not None:
            comp1 = add_c0_to_sersic(update_sersic)
        else:
            comp1 = pyimfit.make_imfit_function('Sersic_GenEllipse')
            comp1.PA.setValue(gal_pa, [-180, 180])
            if gal_e <= 0.6:
                comp1.ell.setValue(0.6, [0.5, 0.99])
            else:
                comp1.ell.setValue(gal_e, [0.05, 0.99])
            comp1.I_e.setValue(cen_flux * 2.0, [cen_flux / 10.0, cen_flux * 200.0])
            comp1.r_e.setValue(gal_a, [0.5, 100])
            comp1.n.setValue(1.2, [0.2, 6.0])
            comp1.c0.setValue(-0.1, [-5, 5])
    else:
        raise ValueError("# Wrong model type!")
        
    galaxy_desc.addFunction(comp1)
    
    # Add optional second model
    if model_type_2 is not None:
        if model_type_2 == 'Sersic':
            comp2 = pyimfit.make_imfit_function('Sersic')
            comp2.PA.setValue(gal_pa, [-180, 180])
            if gal_e >= 0.3:
                comp2.ell.setValue(0.3, [0.02, 0.99])
            else:
                comp2.ell.setValue(gal_e, [0.02, 0.99])
            comp2.I_e.setValue(cen_flux * 2., [cen_flux / 20.0, cen_flux * 100.0])
            comp2.r_e.setValue(4.0, [0.2, 100])
            comp2.n.setValue(3.0, [0.2, 6.0])
        elif model_type_2 == 'Sersic_GenEllipse':
            comp2 = pyimfit.make_imfit_function('Sersic_GenEllipse')
            comp2.PA.setValue(gal_pa, [-180, 180])
            if gal_e >= 0.3:
                comp2.ell.setValue(0.3, [0.02, 0.99])
            else:
                comp2.ell.setValue(gal_e, [0.02, 0.99])
            comp2.I_e.setValue(cen_flux * 2., [cen_flux / 20.0, cen_flux * 100.0])
            comp2.r_e.setValue(4.0, [0.2, 100])
            comp2.n.setValue(3.0, [0.2, 6.0])
            comp2.c0.setValue(0, [-10, 10])
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
        print("# Chi2 statistics: {:f}".format(galaxy_model.fitStatistic))
        print("# Reduced Chi2 statistics: {:f}".format(galaxy_model.reducedFitStatistic))
        print("# AIC statistics: {:f}".format(galaxy_model.AIC))
        print("# BIC statistics: {:f}".format(galaxy_model.BIC))
        print("\n # Best fit parameter values:")
        if solver == 'LM':
            for name, val, err in zip(
                galaxy_model.numberedParameterNames, galaxy_model.getRawParameters(), 
                galaxy_model.getParameterErrors()): 
                print("   {:7s}: {:f}+/-{:f}".format(name, val, err))
        else:
            for name, val in zip(
                galaxy_model.numberedParameterNames, galaxy_model.getRawParameters()): 
                print("   {:7s}: {:f}".format(name, val))

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
            fig.subplots_adjust(left=0.0, right=1.0, bottom=0.1, top=0.99, wspace=0.0, hspace=0.0)

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
    else:
        print("# Model is not converged! Please try again!")
    
    return galaxy_model

#%% STARTING PARAMETERS
test_dir = '/Users/colemeyer/Documents/Isophotal Twist/'

print("# Galaxy 1")
galaxy1 = imfit_prepare_data(
    os.path.join(test_dir, 'AGalaxy1.fits'), os.path.join(test_dir, 'CGalaxy1.fits'), verbose=True)

print("\n# Galaxy 2")
galaxy2 = imfit_prepare_data(
    os.path.join(test_dir, 'AGalaxy2.fits'), os.path.join(test_dir, 'CGalaxy2.fits'), verbose=True)

print("\n# Galaxy 3")
galaxy3 = imfit_prepare_data(
    os.path.join(test_dir, 'AGalaxy3.fits'), os.path.join(test_dir, 'CGalaxy3.fits'), verbose=True)

print("\n# Galaxy 4")
galaxy4 = imfit_prepare_data(
    os.path.join(test_dir, 'AGalaxy4.fits'), os.path.join(test_dir, 'CGalaxy4.fits'), verbose=True)

galaxies = (galaxy1,galaxy2,galaxy3,galaxy4)

#%% RUN MODELS
galaxy = galaxy4

print("\n# 2-D Sersic model:")
sersicfit_model_1 = imfit_fit_edgeon(galaxy, model_type='Sersic', solver='LM')

print("\n# Generalized 2-D Sersic model:")
sersicfit_model_2 = imfit_fit_edgeon(galaxy, model_type='Sersic_GenEllipse', solver='LM',
                                    update_sersic=sersicfit_model_1)

sersicfit_new = update_galaxy_geometry(galaxy, sersicfit_model_2, model_type='Sersic')

print("\n# Double Sersic components model:")
sersicfit_model_3 = imfit_fit_edgeon(
 sersicfit_new, model_type='Sersic', solver='LM', model_type_2='Sersic_GenEllipse')