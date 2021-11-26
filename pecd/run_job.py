import numpy as np
import os
import subprocess
import CONSTANTS
import json
import itertools
import importlib
import time

def convert(o):
    if isinstance(o, np.generic): return o.item()  
    raise TypeError



def gen_euler_grid_2D(n_euler):
    """ Cartesian product of 1D grids of Euler angles"""
    alpha_1d        = list(np.linspace(0, 2*np.pi,  num=1, endpoint=False))
    beta_1d         = list(np.linspace(0, np.pi,    num=n_euler, endpoint=True))
    gamma_1d        = list(np.linspace(0, 2*np.pi,  num=n_euler, endpoint=False))
    euler_grid_3d   = np.array(list(itertools.product(*[alpha_1d, beta_1d, gamma_1d]))) #cartesian product of [alpha,beta,gamma]

    n_euler_3d      = euler_grid_3d.shape[0]
    print("\nTotal number of 2D-Euler grid points: ", n_euler_3d , " and the shape of the 3D grid array is:    ", euler_grid_3d.shape)
    #print(euler_grid_3d)
    return euler_grid_3d, n_euler_3d

def gen_euler_grid(n_euler):
    """ Cartesian product of 1D grids of Euler angles"""
    alpha_1d        = list(np.linspace(0, 2*np.pi,  num=n_euler, endpoint=False))
    beta_1d         = list(np.linspace(0, np.pi,    num=n_euler, endpoint=True))
    gamma_1d        = list(np.linspace(0, 2*np.pi,  num=n_euler, endpoint=False))
    euler_grid_3d   = np.array(list(itertools.product(*[alpha_1d, beta_1d, gamma_1d]))) #cartesian product of [alpha,beta,gamma]

    n_euler_3d      = euler_grid_3d.shape[0]
    print("\nTotal number of 3D-Euler grid points: ", n_euler_3d , " and the shape of the 3D grid array is:    ", euler_grid_3d.shape)
    #print(euler_grid_3d)
    return euler_grid_3d, n_euler_3d

def save_euler_grid(grid_euler, path):   
    with open( path + "grid_euler.dat" , 'w') as eulerfile:   
        np.savetxt(eulerfile, grid_euler, fmt = '%15.4f')

def save_input_file(params,filename):
    with open(params['job_directory']+ "input_"+filename, 'w') as input_file: 
        json.dump(params, input_file, indent=4, default=convert)

def create_dirs(params):

    os.chdir(params['working_dir'])
    path =  params['job_directory']

    isdir = os.path.isdir(path) 
    if isdir:
        print("job directory exists: " + str(isdir) + ", " + path) 
    else:
        print("creating job directory: " + path) 
        os.mkdir(params['job_directory'])
        os.chdir(params['job_directory'])
        os.mkdir("esp")
        os.mkdir("graphics")
    
        os.chdir("graphics")
        os.mkdir("space")
        os.mkdir("momentum")
        os.chdir("../")
        os.chdir("esp")
        for irun in range(params['N_batches']):
            os.mkdir(str(irun))

    os.chdir(params['main_dir'])
    return path


def gen_inputs_list(params_input):

    params_list = []

    """ Generate a grid of molecular orientations parametrized with the Euler angles"""
    if params_input['orient_grid_type'] == "3D":
        grid_euler, params_input['n_grid_euler_3d'] = gen_euler_grid(params_input['N_euler'])            
    elif params_input['orient_grid_type'] == "2D":
        grid_euler, params_input['n_grid_euler_2d'] = gen_euler_grid_2D(params_input['N_euler'])            
    else:
        raise ValueError("incorrect euler grid typ")



    lmin        = params_input['bound_lmax_arr'][0]
    lmax        = params_input['bound_lmax_arr'][1]
    nl          = params_input['bound_lmax_arr'][2]

    nlobmin     = params_input['bound_nlobs_arr'][0]
    nlobmax     = params_input['bound_nlobs_arr'][1]
    nlobattos   = params_input['bound_nlobs_arr'][2]

    rbinmin     = params_input['bound_binw_arr'][0]
    rbinmax     = params_input['bound_binw_arr'][1]
    nrbin       = params_input['bound_binw_arr'][2]

    binrange    = np.linspace(rbinmin,rbinmax,nrbin,endpoint=True,dtype=float)
    lrange      = np.linspace(lmin,lmax,nl,endpoint=True,dtype=int)
    nrange      = np.linspace(nlobmin,nlobmax,nlobattos,endpoint=True,dtype=int)

    
    for l in lrange:
        params_input['bound_lmax'] = l

        for n in nrange:
            params_input['bound_nlobs'] = n

            for r in binrange:
                print("\n\n")
                print("l = " + str(l) + ", n = " + str(n) + ", Rb = " + str(r)+ ":\n")
                params_input['bound_binw'] = r

                params_list.append(setup_input(params_input))
               
    return params_list, grid_euler

def field_params(params):
    """ Define field parameters"""

    time_to_au  = CONSTANTS.time_to_au[ params['time_units'] ]

    """==== field dictionaries ===="""
    field_dict = {}
    env_dict = {}

    if  params['field_func_name']  == "RCPL":
        field_dict   = {
                        "function_name": "fieldRCPL",
                        "omega":            params['omega'], 
                        "E0":               params['E0'], 
                        "CEP0":             params['CEP0'], 
                        "spherical":        True}
    elif params['field_func_name'] == "LCPL":         
        field_dict  = { 
                        "function_name": "fieldLCPL",
                        "omega":            params['omega'], 
                        "E0":               params['E0'], 
                        "CEP0":             params['CEP0'], 
                        "spherical":        True}
    elif params['field_func_name'] == "LP":   
        field_dict     = { 
                            "function_name": "fieldLP",
                            "omega":            params['omega'], 
                            "E0":               params['E0'], 
                            "CEP0":             params['CEP0']}
    else:
        raise ValueError("Incorrect field name")

    # if gaussian width is given as: e^-t^2/sigma^2
    # FWHM = 2.355 * sigma/sqrt(2)


    if params['field_env_name'] == "gaussian":
        env_dict = {
                        "function_name": "envgaussian", 
                        "FWHM": 2.355 * (time_to_au * params['gauss_tau'])/np.sqrt(2.0), 
                        "t0": (time_to_au * params['gauss_t0'])  }

    elif params['field_env_name'] == "sin2":
        env_dict = {
                        "function_name": "envsin2", 
                        "Ncycles": params['sin2_ncycles'] , 
                        "t0": (time_to_au * params['sin2_t0']),
                        "t_cycle": params['opt_cycle']  }

    return field_dict, env_dict

def setup_input(params_input):
    """ Note: All quantities are converted to atomic units"""

    params = {}
    params.update(params_input)

    """ === molecule directory ==== """ 
    if params_input['jobtype'] == "slurm":
        params['main_dir']      = "/beegfs/desy/group/cfel/cmi/zakemil/PECD_personal/PECD/pecd/" 
        params['working_dir']   = "/beegfs/desy/group/cfel/cmi/zakemil/PECD_personal/PECD/tests/molecules/" + params['molec_name'] + "/"
    elif params_input['jobtype'] == "local":
        params['main_dir']      = "/home/emil/Desktop/projects/PECD_personal/PECD/pecd/"
        params['working_dir']   = "/home/emil/Desktop/projects/PECD_personal/PECD/tests/molecules/" + params['molec_name'] + "/"

    """ !!!In this version we are using uniform sized bins, allowing for an array of jobs
        Later expand to general radial basis defined by FEMLIST + array of input parameters!!!"""

    #params['bound_nbins'], params['bound_nlobs'], params['bound_binw'] =  params_input['bound_nbins'], params_input['bound_nlobs'], params_input['bound_binw']
    #params['nbins'], params['nlobs'], params['binw'] = params_input['bound_nbins'], params_input['bound_nlobs'], params_input['bound_binw']
    #params['nbins_cont'] = 0

    """ list defining the radial grid"""


    params['FEMLIST']   = [     [params['bound_nbins'], params['bound_nlobs'], params['bound_binw']] ,\
                                [0, params['bound_nlobs'], params['bound_binw']] ] 

    params['FEMLIST_PROP']   = [ [params['prop_nbins'], params['bound_nlobs'], params['bound_binw']] ,\
                                [0, params['bound_nlobs'], params['bound_binw']] ] 


    params['job_directory'] =  params['working_dir'] + params['molec_name']   + \
                                "_" + str(params['bound_lmax'])    + \
                                "_" + str(params['bound_nlobs'])   + \
                                "_" + str('{:4.2f}'.format(params['bound_binw']))    + \
                                "_" + str(params['bound_nbins'])   + \
                                "_" + str(params['job_label']) +"/"


    if params['mode'] == 'propagate':


        params['rot_wf_file']       = params['working_dir'] + "rv_wavepackets/" + "wavepacket_J60.h5"
        params['rot_coeffs_file']   = params['working_dir'] + "rv_wavepackets/" + "coefficients_j0_j60.rchm"

        params['wavepacket_file']    = "wavepacket"


        """==== file paths and names ===="""


        params['file_psi0']         =   "psi0_" + params['molec_name']   + \
                                        "_" + str(params['bound_lmax'])    + \
                                        "_" + str(params['bound_nlobs'])   + \
                                        "_" + str('{:4.2f}'.format(params['bound_binw']))   + \
                                        "_" + str(params['bound_nbins'])   + \
                                        "_" + str(params['job_label'])   + ".dat"

        params['file_hmat0']        =   "hmat0_" + params['molec_name']   + \
                                        "_" + str(params['bound_lmax'])    + \
                                        "_" + str(params['bound_nlobs'])   + \
                                        "_" + str('{:4.2f}'.format(params['bound_binw']))    + \
                                        "_" + str(params['bound_nbins'])   + \
                                        "_" + str(params['job_label'])   + ".dat"

        params['file_enr0']         =   "enr0_" + params['molec_name']   + \
                                        "_" + str(params['bound_lmax'])    + \
                                        "_" + str(params['bound_nlobs'])   + \
                                        "_" + str('{:4.2f}'.format(params['bound_binw']))   + \
                                        "_" + str(params['bound_nbins'])   + \
                                        "_" + str(params['job_label'])   + ".dat"



        params['file_esp']          =   "esp_" + params['molec_name']   + \
                                        "_" + str(params['bound_lmax'])    + \
                                        "_" + str(params['bound_nlobs'])   + \
                                        "_" + str('{:4.2f}'.format(params['bound_binw']))    + \
                                        "_" + str(params['bound_nbins'])   + \
                                        "_" + str(params['job_label'])    + ".dat"



        params['file_quad_levels']  =   "quad_levels_" + params['molec_name']   + \
                                        "_" + str(params['bound_lmax'])    + \
                                        "_" + str(params['bound_nlobs'])   + \
                                        "_" + str('{:4.2f}'.format(params['bound_binw']))    + \
                                        "_" + str(params['bound_nbins'])   + \
                                        "_" + str(params['job_label'])  + \
                                        "_" + str(params['sph_quad_tol'])   + ".dat"
        
        """ ******** Field frequency *********"""
        if params['freq_units'] == "nm":
            params['omega']     = 10**9 *  CONSTANTS.vellgt / params_input['omega'] # from wavelength (nm) to frequency  (Hz)
        elif params['freq_units'] == "ev":
            params['omega']     = CONSTANTS.ev_to_hz * params_input['omega']  # from ev to frequency  (Hz)
        else:
            raise ValueError("Incorrect units for frequency")

        opt_cycle_as                 = 1.0e18/params['omega'] #in as
        suggested_no_pts_per_cycle = 50     # time-step can be estimated based on the carrier frequency of the pulse. 
                                            # Guan et al. use 1000 time-steps per optical cycle (in small Krylov basis). 
                                            # We can use much less. Demekhin used 50 pts/cycle.
                                            # 1050 nm = 1.179 eV = 285 THz -> 1 optical cycle = 3.5 fs
                                            
        print( "Electric field carrier frequency = " + str("%10.3f"%( params['omega'] * 1.0e-12 )) + " THz" )
        print( "Electric field oscillation period (optical cycle) = " + str("%10.3f"%(1.0e15/params['omega'])) + " fs")
        print( "Suggested time-step (" + str(suggested_no_pts_per_cycle) + "pts/cycle) for field linear frequency = " + str("%12.3f"%(params['omega']/1e12)) + " THz is: " + str("%10.2f"%(opt_cycle_as/suggested_no_pts_per_cycle )) + " as")

        params['omega']     *= 2.0 * np.pi                      # linear to angular frequency
        params['omega']     *= CONSTANTS.freq_to_au['Hz']       # Hz to a.u.
        params['opt_cycle'] = 2.0 * np.pi /params['omega']  # optical cycle (a.u.)

        #params['E0'] = 1.0e9 #V/cm
        #field strength in a.u. (1a.u. = 5.1422e9 V/cm). For instance: 5e8 V/cm = 3.3e14 W/cm^2
        #convert from W/cm^2 to V/cm

        """ ******** Field strength *********"""
        field_units     = "V/cm"
        field_strength  = np.sqrt(params_input['intensity']/(CONSTANTS.vellgt * CONSTANTS.epsilon0))
        print("field strength = " + "  %8.2e"%field_strength + " " + field_units)

        params['E0']        = field_strength
        params['E0']        *= CONSTANTS.field_to_au[field_units] 



        """ ******** Create field dictionaries *********"""
        params['field_type'], params['field_env'] = field_params(params)  


    return params

def run_array_job(params_list,grid_euler):

    for iparams in params_list:

        """ Create directories """
        path = create_dirs(iparams)

        if iparams['mode'] == 'propagate':
            """ Save input file and euler angles grid """
            print("mode = propagate")
            save_input_file(iparams,"prop")
            save_euler_grid(grid_euler, path)

        elif iparams['mode'] == 'analyze':
            print("mode = analyze")
            save_input_file(iparams,"analyze")
        else:
            raise ValueError("incorrect mode")


        """ Run batches """

        if iparams['jobtype'] == "slurm":
            flag = []
            print("Submitting a SLURM job")
            path = os.getcwd()
            print ("The current working directory is %s" % path)
            START_FILES = os.listdir(path + "/slurm_run")
            os.chdir(path+"/slurm_run")
            
            print ("Job directory is %s" % iparams['job_directory'])

            for ibatch in range(iparams['N_batches']):
                time.sleep(2)
                if iparams['mode'] == "propagate":
                    pecd_process =  "./master_script.sh " + str(ibatch) +\
                                    " " + str(iparams['job_directory']) + " " +\
                                    str("PROPAGATE.py")
                    iflag = subprocess.call(pecd_process, shell=True)
                    flag.append([ibatch,iflag])

                elif iparams['mode'] == "analyze":
                    pecd_process =  "./master_script.sh " + str(ibatch) +\
                                    " " + str(iparams['job_directory']) + " " +\
                                    str("ANALYZE.py")
                    iflag = subprocess.call(pecd_process, shell=True)
                    flag.append([ibatch,iflag])

            print("Termination flags for euler grid array job: [ibatch,flag]")        
            print(flag)

        elif iparams['jobtype'] == "local":
            flag = []
            print("Executing local job")
            
            path = os.getcwd()
            print ("The current working directory is %s" % path)
            print ("Job directory is %s" % iparams['job_directory'])
            print("Number of batches = " + str(iparams['N_batches']))

            for ibatch in range(0,iparams['N_batches']):
                if iparams['mode'] == "propagate":
                    pecd_process = "python3 PROPAGATE.py " 	+ str(ibatch)  + " " +str(iparams['job_directory'])
                    iflag = subprocess.call(pecd_process, shell=True) 
                    flag.append([ibatch,iflag])

                elif iparams['mode'] == "analyze":
                    pecd_process = "python3 ANALYZE.py " 	+ str(ibatch)  + " " +str(iparams['job_directory'])
                    iflag = subprocess.call(pecd_process, shell=True) 
                    flag.append([ibatch,iflag])

            print("Termination flags for euler grid array job: [ibatch,flag]")
            print(flag)




if __name__ == "__main__":    


    inputfile 	= "input_chiralium" #input file name
    input_module = importlib.import_module(inputfile)

    print("input file: " + str(inputfile))

    params_input = input_module.read_input()
    print("jobtype: " + str(params_input['jobtype']))

    params_list, grid_euler = gen_inputs_list(params_input)

    run_array_job(params_list,grid_euler)