import numpy as np
import constants
import os
def gen_input():

    params = {}

    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    """ === molecule directory ==== """ 
    """ in this directory we read/write files associated with a given molecule """
    params['main_dir']      = "/Users/zakemil/Nextcloud/projects/PECD/pecd/"
    params['working_dir']   = "/Users/zakemil/Nextcloud/projects/PECD/tests/molecules/h2o/"
    params['molec_name']    = "h2o"


    """==== BOUND ===="""

    params['map_type'] = 'DVR' #DVR or SPECT
    params['hmat_format']   = "regular" # regular, coo, csr

    """==== basis set parameters for BOUND ===="""

    params['bound_nlobs']   = 8
    params['bound_nbins']   = 1
    params['bound_binw']    = 20.0
    params['bound_rshift']  = 0.001 
    params['bound_lmax']    = 4
    
    params['save_ham0']     = True #save the calculated bound state Hamiltonian
    params['save_psi0']     = True #save psi0
    params['save_enr0']     = True #save eigenenergies for psi0

    params['num_ini_vec']   = 10 # number of initial wavefunctions (orbitals) stored in file


    """==== potential energy matrix ===="""
    params['gen_adaptive_quads'] = True
    params['use_adaptive_quads'] = True
    params['sph_quad_global'] = "lebedev_023" #global quadrature scheme in case we don't use adaptive quadratures.
    params['sph_quad_tol']  = 1e-5
    params['calc_method'] = 'jit' #jit, quadpy, vec

    """==== electrostatic potential ===="""
    params['esp_file']          = "esp_grid_h2o_uhf_631Gss_8_0.2_com"
    params['esp_method']        = "uhf_631Gss"
    #"esp_grid_quad_h2o_uhf_631Gss_l10_N40_com"  #filename for the grid representation of the ESP
    params['r_cutoff']          = 8.0 #cut-off radius for the cation electrostatic potential. We are limited by the capabilities of psi4, memory. Common sense says to cut-off the ESP at some range to avoid spurious operations
    """ Note: r_cutoff can be infered from quad_levels file: when matrix elements of esp are nearly an overlap between spherical funcitons, it is good r_in for setting esp=0"""

    params['esp_mode']      = "exact" #exact or interpolate
    params['save_esp_xyzgrid'] = False #generate xyz grid (quadrature grid) for psi4.
    params['plot_esp']      = False

    """==== file paths and names ===="""

    params['file_psi0']     =   "psi0_" + params['molec_name']   + \
                                "_" + str(params['bound_nbins'])   + \
                                "_" + str(params['bound_nlobs']) + \
                                "_" + str(params['bound_binw'])    + \
                                "_" + str(params['bound_lmax'])  + \
                                "_" + params['esp_method']  + ".dat"


    params['file_hmat0']    =   "hmat0_" + params['molec_name']   + \
                                "_" + str(params['bound_nbins'])   + \
                                "_" + str(params['bound_nlobs']) + \
                                "_" + str(params['bound_binw'])    + \
                                "_" + str(params['bound_lmax'])  + \
                                "_" + params['esp_method']  + ".dat"

    params['file_enr0']     =   "enr0_" + params['molec_name']   + \
                                "_" + str(params['bound_nbins'])   + \
                                "_" + str(params['bound_nlobs']) + \
                                "_" + str(params['bound_binw'])    + \
                                "_" + str(params['bound_lmax'])  + \
                                "_" + params['esp_method']   + ".dat"

    params['file_quad_levels']  =   "quad_levels_" + params['molec_name']   + \
                                "_" + str(params['bound_nbins'])   + \
                                "_" + str(params['bound_nlobs']) + \
                                "_" + str(params['bound_binw'])    + \
                                "_" + str(params['bound_lmax'])  + \
                                "_" + "uhf_631Gss" + \
                                "_" + str(params['sph_quad_tol'])   + ".dat"

    return params
