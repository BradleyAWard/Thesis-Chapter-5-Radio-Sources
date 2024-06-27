# --------------------------------------------------
# Imports
# --------------------------------------------------

import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from astropy.constants import c, h, k_B
from lmfit import Model, Parameters
from tqdm import tqdm
from itertools import combinations, zip_longest

# --------------------------------------------------
# Cosmological Functions
# --------------------------------------------------

def separation_to_distance(r, z):
    """
    Converts an angular separation to a physical distance
    
    :param r: Angular separation [arcsec]
    :param z: Redshift
    :return: Separation [kpc]
    """
    r = r*u.arcsec
    d_A = cosmo.angular_diameter_distance(z)
    distance_kpc = (r*d_A).to(u.kpc, u.dimensionless_angles())
    return distance_kpc.value

# --------------------------------------------------

def distance_to_separation(d, z):
    """
    Converts a physical distance to an angular separation
    
    :param d: Separation [kpc]
    :param z: Redshift
    :return: Angular separation [arcsec]
    """
    d = d*u.kpc
    d_A = cosmo.angular_diameter_distance(z)
    r = (d/d_A).to(u.arcsec, u.dimensionless_angles())
    return r.value

# --------------------------------------------------
# Modified Blackbody Functions
# --------------------------------------------------

def bb(nu, t):
    """
    Returns a blackbody
    
    :param nu: Rest frame frequency
    :param t: Temperature [K]
    :return: Planck function
    """
    return (nu**3)*(1/(np.exp((h.value*nu)/(k_B.value*t)) - 1))

def mbb(nu_rest, log_norm, t, beta):
    """
    Returns a modified blackbody
    
    :param nu_rest: Rest frame frequency
    :param log_norm: Normalization
    :param t: Dust temperature [K]
    :param beta: Dust emissivity spectral index
    :return: Modified Planck function
    """
    norm = 10**log_norm
    return norm*(nu_rest**beta)*bb(nu_rest, t)

# --------------------------------------------------
# Classes for Individual Objects
# --------------------------------------------------

class Object:
    """
    Class for unspecified object with RA and Dec
    """
    def __init__(self, name, ra_deg, dec_deg):
        self.id = name
        self.ra = ra_deg
        self.dec = dec_deg

# --------------------------------------------------

class Source(Object):
    """
    Class for sources with an SED
    """
    def __init__(self, name, ra, dec, wavelengths_obs_um, fluxes, flux_errors):
        Object.__init__(self, name=name, ra_deg=ra, dec_deg=dec)
        self.wavelengths_obs_um = wavelengths_obs_um
        self.fluxes = fluxes
        self.flux_errors = flux_errors

# --------------------------------------------------

class Counterpart(Object):
    """
    Class for a counterpart to a source
    """
    def __init__(self, name, ra, dec, fluxes, redshift, redshift_err, **kwargs):
        Object.__init__(self, name=name, ra_deg=ra, dec_deg=dec)
        self.fluxes = fluxes
        self.redshift = redshift
        self.redshift_err = redshift_err
        self.stellar_mass = kwargs.get('stellar_mass')
        self.age = kwargs.get('age')

    @property
    def luminosity_distance(self):
        """
        Returns the luminosity distance of the counterpart

        :return: Luminosity distance [Mpc]
        """
        return cosmo.luminosity_distance(z = self.redshift)

    def number_density(self, counterparts, area):
        """
        Returns the number density brighter than this counterpart
        
        :param counterparts: Instances of counterpart class
        :param area: Area of survey [arcsec2]
        :return: Number density [arcsec-2]
        """
        return np.sum([counterpart.fluxes > self.fluxes for counterpart in counterparts])/area

    def separation(self, other):
        """
        Calculates the separation between two objects given (RA, Dec)
        
        :param other: Instance of another object
        :return: Separation between objects [arcsec]
        """
        return np.sqrt(((self.ra - other.ra)**2)+((self.dec - other.dec)**2))*3600

    def s_value(self, source, counterparts, area):
        """
        Calculates the S value for the counterpart
        
        :param source: The source object this counterpart relates to
        :param counterparts: All other possible counterparts
        :param area: Area of survey [arcsec2]
        :return: S value
        """
        r = self.separation(source)
        n = self.number_density(counterparts, area)
        s = (r**2)*n
        return s

# --------------------------------------------------
# Create Random Sources
# --------------------------------------------------

def random_sources(n, ra_min, ra_max, dec_min, dec_max):
    """
    Creates a set of random sources
    
    :param n: Number of random sources
    :param ra_min: Minimum RA
    :param ra_max: Maximum RA
    :param dec_min: Minimum Dec
    :param dec_max: Maximum Dec
    :return: List of instances of random sources
    """
    sources_rand = []
    for it in range(n):
        # Randomly select RA and Dec and create a source instance
        ra_random, dec_random = np.random.uniform(ra_min, ra_max), np.random.uniform(dec_min, dec_max)
        random_source = Source(it, ra_random, dec_random, np.array([0]), np.array([0]), np.array([0]))
        sources_rand.append(random_source)
    return sources_rand

# --------------------------------------------------
# Classes for Groups of Objects
# --------------------------------------------------

class Pair:
    """
    Class for a pair of objects (source and counterpart)
    """
    def __init__(self, source, counterpart, r, s, p, fit_sed=True):
        self.source = source
        self.counterpart = counterpart
        self.r = r
        self.s = s
        self.p = p

        self.t_cmb = 2.725*(1+self.counterpart.redshift)
        self.source.wavelengths_obs_m = self.source.wavelengths_obs_um*1e-6
        self.source.wavelengths_rest_m = self.source.wavelengths_obs_m/(1+self.counterpart.redshift)
        self.source.nu_rest = c.value/self.source.wavelengths_rest_m

        # Fitting a one temperature SED
        if fit_sed:
            
            # If no redshift given, do not return a fit
            if np.isnan(self.counterpart.redshift):
                self.log_norm = np.nan
                self.t = np.nan
                self.beta = np.nan
            else:
                # Fitting an optically thin model with free normalization, temperature and beta
                mbb_model = Model(mbb)
                params = Parameters()
                params.add_many(('log_norm', -60, True, -65, -55),
                                ('t', 20, True, self.t_cmb, 50),
                                ('beta', 2, True, 1, 4))
                mbb_fit = mbb_model.fit(self.source.fluxes, params, nu_rest=self.source.nu_rest, weights=1/self.source.flux_errors)

                # Obtain best fitting parameters
                self.log_norm = mbb_fit.params['log_norm'].value
                self.t = mbb_fit.params['t'].value
                self.beta = mbb_fit.params['beta'].value

    def sed(self, nu_rest):
        """
        Returns an SED for the source
        
        :param nu_rest: Rest frame frequency
        :return: SED [Jy]
        """
        return mbb(nu_rest, self.log_norm, self.t, self.beta)*u.Jy

    def ir_luminosity(self, lam_um_rest_low=8, lam_um_rest_high=1000):
        """
        Calculates the bolometric IR luminosity of the source
        
        :param lam_um_rest_low: Low rest frame wavelength for integration [microns] (Default = 8)
        :param lam_um_rest_high: High rest frame wavelength for integration [microns] (Default = 1000)
        :return: IR luminosity [Lsun]
        """
        # Define rest frame frequencies and select integration range
        lam_low_um_rest, lam_high_um_rest = lam_um_rest_low*u.micron, lam_um_rest_high*u.micron
        wave_range_rest_um = np.linspace(1, 5000, 100000) * u.micron
        wave_range_rest_m = wave_range_rest_um.to(u.m)
        freq_range_rest = c / wave_range_rest_m
        idx = np.where((wave_range_rest_um >= lam_low_um_rest) & (wave_range_rest_um <= lam_high_um_rest))

        # Calculate the frequency gradient across the SED
        diff_freq = np.diff(freq_range_rest)
        diff_freq = np.append(diff_freq, diff_freq[-1])

        # Integrate the SED between limits
        sed_obs_integral = self.sed(freq_range_rest[idx].value)
        sed_rest_integral = sed_obs_integral/(1+self.counterpart.redshift)
        integral = np.sum(-sed_rest_integral*diff_freq[idx])

        # Convert integral into a bolometric luminosity
        d_L = cosmo.luminosity_distance(z=self.counterpart.redshift).to(u.m)
        l_watt = (4 * np.pi * (d_L ** 2) * integral).to(u.Watt)
        l_sun = l_watt.to(u.L_sun)
        return l_sun

    def sfr(self):
        """
        Calculates the Star Formation Rate from Murphy et al., 2011
        
        :return: SFR [Msun yr-1]
        """
        l_sun = self.ir_luminosity()
        l_cgs = l_sun.to(u.erg/u.s)
        constant = 3.88e-44*(u.M_sun/u.yr)/(u.erg/u.s)
        sfr = constant*l_cgs
        return sfr

# --------------------------------------------------

class Group:
    """
    Class for a group of objects (pairs with a common source)
    """
    def __init__(self, pairs):
        self.pairs = pairs
        self.counterparts = [pair.counterpart for pair in self.pairs]
        self.source = [pair.source for pair in self.pairs][0]

    @property
    def average_counterpart_source_separation(self):
        """
        Determines the average separation between source and counterpart
        
        :return: Separation [arcsec]
        """
        # Determine the average RA and Dec of counterparts
        av_counterpart_ra = np.mean([pair.counterpart.ra for pair in self.pairs])
        av_counterpart_dec = np.mean([pair.counterpart.dec for pair in self.pairs])

        # Determine the source RA, Dec and calculate separation
        source_ra, source_dec = self.pairs[0].source.ra, self.pairs[0].source.dec
        separation = np.sqrt(((source_ra - av_counterpart_ra)**2)+((source_dec - av_counterpart_dec)**2))*3600
        return separation

    @property
    def counterpart_flux_sorted(self, n_max=3):
        """
        Sort the counterpart flux densities
        
        :param n_max: Maximum number of counterparts
        :return: Sorted counterpart flux densities
        """
        fluxes = sorted([obj.fluxes.item() for obj in self.counterparts], reverse=True)
        n = len(fluxes)
        return np.pad(fluxes, (0,n_max-n), 'constant', constant_values=np.nan)

    def counterpart_source_separations(self):
        """
        Determines the separation between source and counterparts

        :return: List of separations [arcsec]
        """
        separations_arcsec = [pair.r for pair in self.pairs]
        return separations_arcsec

    def counterpart_counterpart_separations(self):
        """
        Determines the separation between counterparts
        
        :return: List of separations [arcsec]
        """
        # Identify all counterpart pairs
        obj_list = list(range(len(self.pairs)))
        index_list = list(combinations(obj_list,2))

        # Calculate separations
        separations_arcsec = [self.pairs[it[0]].counterpart.separation(self.pairs[it[1]].counterpart) for it in index_list]
        return separations_arcsec
    
    def counterpart_counterpart_separations_new(self):
        """
        Determines the separation between counterparts
        
        :return: List of separations [arcsec]
        """
        # Identify all counterpart pairs
        obj_list = list(range(len(self.pairs)))
        index_list = list(combinations(obj_list,2))

        # Calculate separations
        separations_arcsec_ra = [(self.pairs[it[0]].counterpart.ra - self.pairs[it[1]].counterpart.ra)*3600 for it in index_list]
        separations_arcsec_dec = [(self.pairs[it[0]].counterpart.dec - self.pairs[it[1]].counterpart.dec)*3600 for it in index_list]
        return separations_arcsec_ra, separations_arcsec_dec

    def counterpart_counterpart_z_difference(self):
        """
        Determines the separation of redshifts between counterparts
        
        :return: List of redshift differences and their errors
        """
        # Identify all counterpart pairs
        obj_list = list(range(len(self.pairs)))
        index_list = list(combinations(obj_list,2))

        # Calculate redshift differences and errors
        z_diff = [abs(self.pairs[it[0]].counterpart.redshift - self.pairs[it[1]].counterpart.redshift) for it in index_list]
        z_diff_err = [np.sqrt((self.pairs[it[0]].counterpart.redshift_err**2) + (self.pairs[it[1]].counterpart.redshift_err**2)) for it in index_list]
        return z_diff, z_diff_err

    def counterpart_counterpart_z_difference_new(self):
        """
        Determines the separation of redshifts between counterparts
        
        :return: List of redshift differences and their errors
        """
        # Identify all counterpart pairs
        obj_list = list(range(len(self.pairs)))
        index_list = list(combinations(obj_list,2))

        # Calculate redshift differences and errors
        z_diff = [self.pairs[it[0]].counterpart.redshift - self.pairs[it[1]].counterpart.redshift for it in index_list]
        z_diff_err = [np.sqrt((self.pairs[it[0]].counterpart.redshift_err**2) + (self.pairs[it[1]].counterpart.redshift_err**2)) for it in index_list]
        return z_diff, z_diff_err

    def counterpart_flux_contribution(self):
        """
        Determines the counterpart contributions to total flux
        
        :return: List of flux density contributions
        """
        fluxes = self.counterpart_flux_sorted
        total_flux = np.nansum(fluxes)
        contribution = fluxes/total_flux
        return contribution
    
# --------------------------------------------------

class Survey:
    """
    Class for a survey of sources and counterparts
    """
    def __init__(self, sources, counterparts):

        # Input sources and counterparts
        self.sources_all = sources
        self.counterparts_all = counterparts
        print('Input Number of Sources = {}'.format(len(self.sources_all)))
        print('Input Number of Counterparts = {}'.format(len(self.counterparts_all)))

        # Define the overlapping survey area
        sources_ra = [self.sources_all[obj].ra for obj in range(len(self.sources_all))]
        sources_dec = [self.sources_all[obj].dec for obj in range(len(self.sources_all))]
        counterparts_ra = [self.counterparts_all[obj].ra for obj in range(len(self.counterparts_all))]
        counterparts_dec = [self.counterparts_all[obj].dec for obj in range(len(self.counterparts_all))]
        self.ra_min, self.ra_max = max(min(sources_ra),min(counterparts_ra)), min(max(sources_ra),max(counterparts_ra))
        self.dec_min, self.dec_max = max(min(sources_dec),min(counterparts_dec)), min(max(sources_dec),max(counterparts_dec))
        self.area_deg = (abs(self.ra_max - self.ra_min)) * (abs(self.dec_max - self.dec_min))
        self.area_arcsec = (np.sqrt(self.area_deg)*3600)**2

        # Reduce sources and counterparts to overlapping region
        self.sources = [obj for obj in self.sources_all if (obj.ra > self.ra_min) & (obj.ra < self.ra_max) & (obj.dec > self.dec_min) & (obj.dec < self.dec_max)]
        self.counterparts = [obj for obj in self.counterparts_all if (obj.ra > self.ra_min) & (obj.ra < self.ra_max) & (obj.dec > self.dec_min) & (obj.dec < self.dec_max)]
        print('Overlapping Number of Sources = {} ({:.2f}%)'.format(len(self.sources), (len(self.sources)/len(self.sources_all))*100))
        print('Overlapping Number of Counterparts = {} ({:.2f}%)'.format(len(self.counterparts), (len(self.counterparts)/len(self.counterparts_all))*100))

    def get_random_sources(self, n):
        """
        Generates a set of random sources for the survey

        :param n: The number of randomly created sources
        :return: List of n randomly located sources
        """
        sources_rand = random_sources(n, self.ra_min, self.ra_max, self.dec_min, self.dec_max)
        return sources_rand

    def get_s_values(self, r_max_arcsec, n=100000, random=False, disable=False):
        """
        Calculates the S values for all potential candidates (S-statistic as presented in Dye et al., 2009)

        :param r_max_arcsec: The maximum search radius [arcsec]
        :param n: The number of randomly created sources (Default = 100,000)
        :param random: Boolean for random sources (Default = False)
        :param disable: Disable progress bar (Default = False)
        :return: Minimum S-values for each position
        """

        # Get sources from initialization or produce random sources
        if random:
            sources = self.get_random_sources(n)
        else:
            sources = self.sources

        # Calculating minimum S value for each source
        min_s_values = []
        for source in tqdm(sources, desc = 'Calculating S Values', disable=disable):
            
            # Locating all potential candidates within a given radius
            counterpart_s_values = []
            r_max_deg = r_max_arcsec/3600
            ra_min, ra_max, dec_min, dec_max = source.ra-r_max_deg, source.ra+r_max_deg, source.dec-r_max_deg, source.dec+r_max_deg
            possible_counterparts = [counterpart for counterpart in self.counterparts if (ra_min < counterpart.ra < ra_max) & (dec_min < counterpart.dec < dec_max)]
            
            # For each possible candidate, calculate S and find the minimum value
            for counterpart in possible_counterparts:
                r = counterpart.separation(source)
                if r <= r_max_arcsec:
                    s = counterpart.s_value(source, self.counterparts, self.area_arcsec)
                    counterpart_s_values.append(s)
            min_s = min(counterpart_s_values, default=np.nan)
            min_s_values.append(min_s)
        
        # Remove non-finite values from final array
        min_s_values_finite = np.array([s for s in min_s_values if np.isfinite(s)])
        min_s_values_finite_sorted = sorted(min_s_values_finite)
        return min_s_values_finite_sorted

    def get_matches(self, r_max_arcsec, p_max, n=100000, disable=False):
        """
        Return all counterpart matches within a search radius

        :param r_max_arcsec: The maximum search radius [arcsec]
        :param p_max: The maximum P value for association
        :param n: The number of randomly created sources (Default = 100,000)
        :param disable: Disable progress bar (Default = False)
        :return: Random S value distribution and objects satisfying the maximum P criteria
        """

        # Get random S-values for comparison
        d_s_random = self.get_s_values(r_max_arcsec=r_max_arcsec, n=n, random=True, disable=disable)
        survey_pairs = []
        for source in tqdm(self.sources, desc='Calculating P Values', disable=disable):
            
            # For each source locate all possible candidates
            source_pairs = []
            r_max_deg = r_max_arcsec/3600
            ra_min, ra_max, dec_min, dec_max = source.ra-r_max_deg, source.ra+r_max_deg, source.dec-r_max_deg, source.dec+r_max_deg
            possible_counterparts = [counterpart for counterpart in self.counterparts if (ra_min < counterpart.ra < ra_max) & (dec_min < counterpart.dec < dec_max)]
            
            # For each candidate determine the separation, S value and P value
            for counterpart in possible_counterparts:
                r = counterpart.separation(source)
                if r <= r_max_arcsec:
                    s_i = counterpart.s_value(source, self.counterparts, self.area_arcsec)
                    p = len([s for s in d_s_random if s < s_i])/n

                    # If P is less than 1/N then replace with 1/N
                    if p < 1/n:
                        p = 1/n

                    # If P is less than maximum P then add the source and counterpart as a pair
                    if p <= p_max:
                        pair = Pair(source, counterpart, r, s_i, p)
                        source_pairs.append(pair)
            
            # Sort pairs of sources and counterparts by their P value
            source_pairs_sorted = sorted(source_pairs, key = lambda pair: pair.p)
            survey_pairs.append(source_pairs_sorted)
        return d_s_random, survey_pairs

    def get_groups(self, r_max_arcsec, p_max, n=100000, disable=False):
        """
        Generates groups of sources based on the number of IDs found

        :param r_max_arcsec: The maximum search radius [arcsec]
        :param p_max: The maximum P value for association
        :param n: The number of randomly created sources (Default = 100,000)
        :param disable: Disable progress bar (Default = False)
        :return: Dictionary of groups with different numbers of IDs
        """
        # Get pairs of sources and counterparts ordered by P value
        d_s_random, survey_pairs = self.get_matches(r_max_arcsec=r_max_arcsec, p_max=p_max, n=n, disable=disable)
        survey_array = np.array(list(zip_longest(*survey_pairs, fillvalue=np.nan))).T

        # Identify sources with no IDs
        try:
            blanks_idx = [idx for idx,pairs in enumerate(survey_array) if pd.isna([pairs[0]])]
            blanks_sources = [self.sources[i] for i in blanks_idx]
        except:
            blanks_sources = []
            print('No Blank Sources')

        # Identify sources with one ID
        try:
            singles = survey_array[pd.isna(survey_array[:, 1])][:,0]
            singles_pairs = [pair for pair in singles if ~pd.isna([pair])]
        except:
            singles_pairs = []
            print('No Single Sources')

        # Identify sources with multiple IDs
        try:
            multiples = survey_array[~pd.isna(survey_array[:,1])]
            multiples_list = [multiples[it][~pd.isna(multiples[it])].tolist() for it in range(len(multiples))]
            multiples_group = [Group(pairs=pairs) for pairs in multiples_list]
        except:
            multiples_group = []
            print('No Multiple Sources')

        # Identify primary (lowest P) counterparts
        try:
            primaries_list = [pairs[0] for pairs in survey_array if ~pd.isna([pairs[0]])]
        except:
            primaries_list = []
            print('No Primary Objects')

        # Identify secondary counterparts
        try:
            secondaries_list = [pairs[1] for pairs in survey_array if ~pd.isna([pairs[1]])]
        except:
            secondaries_list = []
            print('No Secondary Objects')

        # Identify tertiary counterparts
        try:
            tertiaries_list = [pairs[2] for pairs in survey_array if ~pd.isna([pairs[2]])]
        except:
            tertiaries_list = []
            print('No Tertiary Objects')

        # Combine all groups above in a dictionary
        groups = {
                'blanks': blanks_sources,
                'singles': singles_pairs,
                'multiples': multiples_group,
                'primaries': primaries_list,
                'secondaries': secondaries_list,
                'tertiaries': tertiaries_list}

        return d_s_random, groups

# --------------------------------------------------
# Apply a flux limit to results
# --------------------------------------------------

def apply_flux_limit(groups, flux_lim, flux_lim_idx=0):
    """
    Restricts a dictionary of groups to above a given flux limit
    
    :param groups: Dictionary of groups as returned from Survey class
    :param flux_lim: Minimum flux [mJy]
    :param flux_lim_idx: Index of wavelength the flux limit applies to
    :return: Dictionary of groups with a minimum flux limit applied
    """
    groups_flux = {}

    # Apply flux limit on blanks, singles, multiples, primaries, secondaries and tertiaries
    groups_flux['blanks'] = [source for source in groups['blanks'] if source.fluxes[flux_lim_idx] > flux_lim]
    groups_flux['singles'] = [pair for pair in groups['singles'] if pair.source.fluxes[flux_lim_idx] > flux_lim]
    groups_flux['multiples'] = [group for group in groups['multiples'] if group.source.fluxes[flux_lim_idx] > flux_lim]
    groups_flux['primaries'] = [pair for pair in groups['primaries'] if pair.source.fluxes[flux_lim_idx] > flux_lim]
    groups_flux['secondaries'] = [pair for pair in groups['secondaries'] if pair.source.fluxes[flux_lim_idx] > flux_lim]
    groups_flux['tertiaries'] = [pair for pair in groups['tertiaries'] if pair.source.fluxes[flux_lim_idx] > flux_lim]
    return groups_flux

# --------------------------------------------------
# Rayleigh Distribution
# --------------------------------------------------

def rayleigh(r, A, sigma):
    """
    Returns a Rayleigh distribution

    :param r: Radial separation [arcsec]
    :param A: Normalization factor
    :param sigma: Positional error [arcsec]
    :return: Rayleigh distribution
    """
    return A*(r/(sigma**2))*(np.exp(-(r**2)/(2*(sigma**2))))