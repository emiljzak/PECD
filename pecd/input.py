import numpy as np
import CONSTANTS
import os
def gen_input():

    params = {}

    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    """ === molecule directory ==== """ 

    params['main_dir']      = "/Users/zakemil/Nextcloud/projects/PECD/pecd/"
    params['working_dir']   = "/Users/zakemil/Nextcloud/projects/PECD/tests/molecules/h2o/"
    params['molec_name']    = "h2o"


    """==== BOUND ===="""

    params['map_type']      = 'DVR' #DVR or SPECT
    params['hmat_format']   = "regular" # regular, coo, csr


    """==== basis set parameters for BOUND ===="""

    params['bound_nlobs']   = 8
    params['bound_nbins']   = 1
    params['bound_binw']    = 20.0
    params['bound_rshift']  = 0.01 
    params['bound_lmax']    = 2
    
    params['save_ham0']     = True #save the calculated bound state Hamiltonian
    params['save_psi0']     = True #save psi0
    params['save_enr0']     = True #save eigenenergies for psi0

    params['num_ini_vec']   = 10 # number of initial wavefunctions (orbitals) stored in file


    """==== potential energy matrix ===="""

    params['gen_adaptive_quads'] = True
    params['use_adaptive_quads'] = True
    params['sph_quad_global']    = "lebedev_023" #global quadrature scheme in case we don't use adaptive quadratures.
    params['sph_quad_tol']       = 1e-5
    params['calc_method']        = 'jit' #jit, quadpy, vec


    """==== electrostatic potential ===="""

    params['esp_method']         = "uhf_631Gss"
    params['esp_mode']           = "exact" #exact or interpolate
    params['plot_esp']           = False
    params['r_cutoff']           = 8.0    

    """ Note: r_cutoff can be infered from quad_levels file: 
                                         when matrix elements of esp are nearly an overlap between spherical funcitons, it is good r_in for setting esp=0.
                                         only in interpolation esp_mode. cut-off radius for the cation electrostatic potential. We are limited by the capabilities of psi4, memory.
                                         Common sense says to cut-off the ESP at some range to avoid spurious operations"""


    """==== file paths and names ===="""

    params['file_psi0']         =   "psi0_" + params['molec_name']   + \
                                    "_" + str(params['bound_nbins'])   + \
                                    "_" + str(params['bound_nlobs']) + \
                                    "_" + str(params['bound_binw'])    + \
                                    "_" + str(params['bound_lmax'])  + \
                                    "_" + str(params['esp_method'])   + ".dat"

    params['file_hmat0']        =   "hmat0_" + params['molec_name']   + \
                                    "_" + str(params['bound_nbins'])   + \
                                    "_" + str(params['bound_nlobs']) + \
                                    "_" + str(params['bound_binw'])    + \
                                    "_" + str(params['bound_lmax'])  + \
                                    "_" + str(params['esp_method'])   + ".dat"

    params['file_enr0']         =   "enr0_" + params['molec_name']   + \
                                    "_" + str(params['bound_nbins'])   + \
                                    "_" + str(params['bound_nlobs']) + \
                                    "_" + str(params['bound_binw'])    + \
                                    "_" + str(params['bound_lmax'])  + \
                                    "_" + str(params['esp_method'])   + ".dat"

    params['file_quad_levels']  =   "quad_levels_" + params['molec_name']   + \
                                    "_" + str(params['bound_nbins'])   + \
                                    "_" + str(params['bound_nlobs']) + \
                                    "_" + str(params['bound_binw'])    + \
                                    "_" + str(params['bound_lmax'])  + \
                                    "_" + str(params['esp_method'])  + \
                                    "_" + str(params['sph_quad_tol'])   + ".dat"

    params['file_esp']          =   "esp_" + params['molec_name']   + \
                                    "_" + str(params['bound_nbins'])   + \
                                    "_" + str(params['bound_nlobs'])   + \
                                    "_" + str(params['bound_binw'])    + \
                                    "_" + str(params['bound_lmax'])    + \
                                    "_" + str(params['esp_method'])    + ".dat"
    

    """==== PROPAGATE ===="""

    params['nlobs']     = 8
    params['nbins']     = 2
    params['binw']      = 20.0

    params['FEMLIST']   = [     [params['bound_nbins'], params['bound_nlobs'],params['bound_binw']] ,\
                                [params['nbins'], params['nlobs'],params['binw']] ] 


    params['t0']        = 0.0 
    params['tmax']      = 300.0 
    params['dt']        = 0.2 
    params['ivec']      = 0

    time_units          = "as"

    params['save_ham_init']      = True #save initial hamiltonian in a file for later use?
    params['save_psi_init']      = True
    params['save_enr_init']      = True
    params['read_ham_init_file'] = True #if available read the prestored initial hamiltonian from file
    
    params['plot_elfield']       = False

    params['wavepacket_file']    = "wavepacket.dat"

    params['file_hmat_init']      =   "hmat_init_" + params['molec_name']   + \
                                    "_" + str(params['bound_nbins'] + params['nbins'])   + \
                                    "_" + str(params['bound_nlobs'] + params['nbins'] * params['nlobs']) + \
                                    "_" + str(params['bound_binw'])    + \
                                    "_" + str(params['bound_lmax'])  + \
                                    "_" + str(params['esp_method'])   + ".dat"

    params['file_psi_init']       =   "psi_init_" + params['molec_name']   + \
                                    "_" + str(params['bound_nbins'] + params['nbins'])   + \
                                    "_" + str(params['bound_nlobs'] + params['nbins'] * params['nlobs']) + \
                                    "_" + str(params['bound_binw'])    + \
                                    "_" + str(params['bound_lmax'])  + \
                                    "_" + str(params['esp_method'])   + ".dat"

    params['file_enr_init']       =   "enr_init_" + params['molec_name']   + \
                                    "_" + str(params['bound_nbins'] + params['nbins'])   + \
                                    "_" + str(params['bound_nlobs'] + params['nbins'] * params['nlobs']) + \
                                    "_" + str(params['bound_binw'])    + \
                                    "_" + str(params['bound_lmax'])  + \
                                    "_" + str(params['esp_method'])   + ".dat"


    """ ====== FIELD PARAMETERS ====== """

    """ ---- carrier frequency ----- """
    params['omega']     = 23.128 
    freq_units          = "nm" #nm or eV

    if freq_units == "nm":
        params['omega']     = 10**9 *  CONSTANTS.vellgt / params['omega'] # from wavelength (nm) to frequency  (Hz)
    elif freq_units == "ev":
        params['omega']     = 0.0   # from ev to frequency  (Hz)
    else:
        raise ValueError("Incorrect units for frequency")

    opt_cycle                  = 1.0e18/params['omega']
    suggested_no_pts_per_cycle = 25     # time-step can be estimated based on the carrier frequency of the pulse. Guan et al. use 1000 time-steps per optical cycle (in small Krylov basis). We can use much less. Demekhin used 50pts/cycle
    # 1050 nm = 1.179 eV = 285 THz -> 1 optical cycle = 3.5 fs
    
    print( "Electric field carrier frequency = " + str("%10.3f"%( params['omega'] * 1.0e-12 )) + " THz" )
    print( "Electric field oscillation period (optical cycle) = " + str("%10.3f"%(1.0e15/params['omega'])) + " fs")
    print( "suggested time-step for field linear frequency = " + str("%12.3f"%(params['omega']/1e12)) + " THz is: " + str("%10.2f"%(opt_cycle/suggested_no_pts_per_cycle )) + " as")

    params['omega'] *= 2.0 * np.pi # linear to angular frequency
    params['omega'] *= CONSTANTS.freq_to_au['Hz'] #Hz to a.u.

    """ ---- field intensity ----- """
    
    #params['E0'] = 1.0e9 #V/cm
    field_units     = "V/cm"
    #field strength in a.u. (1a.u. = 5.1422e9 V/cm). For instance: 5e8 V/cm = 3.3e14 W/cm^2
    #convert from W/cm^2 to V/cm
    intensity       = 7.0e+14 #W/cm^2 #peak intensity
    field_strength  = np.sqrt(intensity/(CONSTANTS.vellgt * CONSTANTS.epsilon0))
    print("field strength = " + "  %8.2e"%field_strength)

    params['E0']        = field_strength
    params['E0']        *= CONSTANTS.field_to_au[field_units] 


       """ ---- field intensity ----- """
    
    params['tau']       = 100.0 #as: pulse duration (sigma)
    params['tc']        = 200.0 #as: pulse centre
    

    """==== field dictionaries ===="""

    field_CPL   = { "function_name": "fieldCPL", 
                    "omega": params['omega'], 
                    "E0": params['E0'], 
                    "CEP0": 0.0, 
                    "spherical": True, 
                    "typef": "LCPL"}

    field_LP    = { "function_name": "fieldLP", 
                    "omega": params['omega'], 
                    "E0": params['E0'], 
                    "CEP0": 0.0}

    # if gaussian width is given: e^-t^2/sigma^2
    # FWHM = 2.355 * sigma/sqrt(2)
    env_gaussian = {"function_name": "envgaussian", 
                    "FWHM": 2.355 * params['tau']/np.sqrt(2.0), 
                    "t0": params['tc']  }

    params['field_form'] = "analytic" #or numerical
    params['field_type'] = field_CPL 

    """ Available field types :
        1) field_CPL
        2) field_LP
        3) field_omega2omega
    """

    params['field_env'] = env_gaussian 

    """ Available envelopes :
        1) env_gaussian
        2) env_flat
    """

    return params
