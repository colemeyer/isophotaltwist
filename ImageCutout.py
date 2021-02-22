import os
import sys

import numpy as np

import astropy.units as u
from astropy import wcs
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.visualization import make_lupton_rgb
from astropy.utils.data import download_file, clear_download_cache

import matplotlib.pyplot as plt

from unagi import config
from unagi import hsc
from unagi import plotting
from unagi.task import hsc_cutout

pdr2 = hsc.Hsc(dr='pdr2', rerun='pdr2_wide')
#pdr2 = hsc.Hsc(dr='dr3', rerun='s20a_dud')

#%% STARTING PARAMETERS
coord_1 = SkyCoord(214.2981528, 52.261633, frame='icrs', unit='deg')
coord_2 = SkyCoord(150.1213, 2.235916, frame='icrs', unit='deg')

# Angular size
s_ang = 10.0 * u.arcsec

# Physical size
s_phy = 200.0 * u.kpc
redshift = 0.42

# Filters
filters = 'i'

# Output dir
output_dir = '/Users/colemeyer/Desktop/' #directory for output
'''
#%% 1. CENTRAL COORDINATE + ANGULAR SIZE
# s_ang represents angle from central coordinate to edge of image
coord_1 = SkyCoord(150.091344, 2.205916, frame='icrs', unit='deg')
s_ang = 10.0 * u.arcsec

cutout_ang = hsc_cutout(coord_1, cutout_size=s_ang, filters=filters, archive=pdr2, 
                         use_saved=False, output_dir=output_dir, verbose=True, 
                         save_output=True)
w = wcs.WCS(cutout_ang[1].header)

_ = plotting.display_single(cutout_ang[1].data)

cutout_ang.close()
'''
#%% 2. CENTRAL COORDINATE + PHYSICAL SIZE
cutout_phy = hsc_cutout(coord_1, cutout_size=s_phy, redshift=redshift, filters=filters, 
                        archive=pdr2, variance=True, mask=True, use_saved=False, output_dir=output_dir, verbose=True, 
                        save_output=True)

_ = plotting.display_single(cutout_phy[1].data)

cutout_phy.close()
'''
#%% 3. DIAGONAL COORDINATES
# Also get the mask and variance plane
# In both g- and i-band
# Save the FITS file with the prefix: "awesome_galaxy"

cutout_multi = hsc_cutout(coord_1, coord_2=coord_2, filters=filters, archive=pdr2, 
                          output_dir=output_dir, variance=True, mask=True, 
                          prefix='awesome_galaxy')

# g-band image
_ = plotting.display_single(cutout_multi[0][1].data)

# i-band variance plane
_ = plotting.display_single(cutout_multi[1][3].data)

#%% EXAMPLE TRI-PLOT
cutout_test = hsc_cutout(coord_1, cutout_size=s_phy, redshift=redshift, filters='i', 
                         archive=pdr2, use_saved=False, output_dir=output_dir, verbose=True, 
                         save_output=True, image=True, mask=True, variance=True)

label_list = [r"$\rm {}$".format(label) for label in ['Image', 'Mask', 'Variance']]
_ = plotting.display_all(
    cutout_test, hdu_list=True, img_size=4, label_list=label_list, fontsize=30, fontcolor='k')
'''