# --------------------------------------------------
# Imports
# --------------------------------------------------

import utils
import numpy as np
from tqdm import tqdm

# --------------------------------------------------
# Create catalogues of sources and counterparts
# --------------------------------------------------

def create_sources_and_counterparts(source_params, counterpart_params):

    source_catalogue = []
    for obj in tqdm(range(len(source_params['data'])), desc='Creating Sources'):
        name = source_params['data'][source_params['name']][obj]
        ra = source_params['data'][source_params['ra']][obj]
        dec = source_params['data'][source_params['dec']][obj]
        wavelengths_obs_um = source_params['wavelengths_obs_um']
        fluxes = np.array(source_params['data'][source_params['fluxes']].iloc[obj])
        flux_errors = np.array(source_params['data'][source_params['flux_errors']].iloc[obj])
        source = utils.Source(name, ra, dec, wavelengths_obs_um, fluxes, flux_errors)
        source_catalogue.append(source)

    counterpart_catalogue = []
    for obj in tqdm(range(len(counterpart_params['data'])), desc='Creating Counterparts'):
        name = counterpart_params['data'][counterpart_params['name']][obj]
        ra = counterpart_params['data'][counterpart_params['ra']][obj]
        dec = counterpart_params['data'][counterpart_params['dec']][obj]
        fluxes = np.array(counterpart_params['data'][counterpart_params['fluxes']].iloc[obj])
        redshift = counterpart_params['data'][counterpart_params['redshift']][obj]
        redshift_err = counterpart_params['data'][counterpart_params['redshift_err']][obj]

        stellar_mass = counterpart_params['data']['lp_mass_med'][obj]
        age = counterpart_params['data']['lp_age'][obj]

        counterpart = utils.Counterpart(name, ra, dec, fluxes, redshift, redshift_err, stellar_mass=stellar_mass, age=age)
        counterpart_catalogue.append(counterpart)

    return source_catalogue, counterpart_catalogue