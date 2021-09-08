import numpy as np
import os
import subprocess

def create_dirs(params):

	os.chdir(params['working_dir'])
	path =  params['job_directory']

	isdir = os.path.isdir(path) 
	if isdir:
		print("job directory exists: " + str(isdir) + ", " + path) 
	else:
		print("creating job directory: " + str(isdir) + ", " + path) 
		os.mkdir(params['job_directory'])
		os.chdir(params['job_directory'])
		os.mkdir("esp")
		os.mkdir("animation")
		os.chdir("esp")
		for irun in range(N_euler_3D):
			os.mkdir(str(irun))
	os.chdir(params['main_dir'])
	return path



def run_array_job(params):

	if jobtype == "slurm":
		flag = []
		print("Submitting SLURM job")
		path = os.getcwd()
		print ("The current working directory is %s" % path)
		START_FILES = os.listdir(path+"/slurm_run")
		os.chdir(path+"/slurm_run")
		print ("Job directory is %s" % path)
		for ibatch in range(N_batches):
			pecd_process = "./master_script.sh " 	+ str(ibatch) 	+\
				 			" " + str(N_batches) + " " + str(N_euler) + " " +\
					 		jobtype + " " + inputfile + " " + jobdir
			iflag = subprocess.call(pecd_process, shell=True)
	
			flag.append([ibatch,iflag])
		print(flag)

	elif jobtype == "local":
		flag = []
		print("Executing local job")
		path = os.getcwd()
		print ("The current working directory is %s" % path)
		for ibatch in range(N_batches):
			iflag = subprocess.call("python3 PROPAGATE.py " 	+ str(ibatch) 	+\
				 			" " + str(N_batches) + " " + str(N_euler) + " "	+\
					 		jobtype + " " + inputfile , shell=True) 
			flag.append([ibatch,iflag])
		print(flag)


def gen_inputs_list(jobtype,params_input):

    params_list = {}

    lmin = params_input['bound_lmax'][0]
    lmax = params_input['bound_lmax'][1]
    nl   = params_input['bound_lmax'][2]


    nlobmin = params_input['bound_nlobs'][0]
    nlobmax  = params_input['bound_nlobs'][1]
    nlobattos = params_input['bound_nlobs'][2]


    rbinmin = params_input['bound_binw'][0]
    rbinmax  = params_input['bound_binw'][1]
    nrbin   = params_input['bound_binw'][2]

    binrange = np.linspace(rbinmin,rbinmax,nrbin,endpoint=True)


    for l in range(lmin,lmax+1,nl):
        print(l)
        params_input['lmax'] = l
        for n in range(nlobmin,nlobmax,nlobattos):
            params_input['nlobatto'] = n
            for r in binrange:
                print(r)
                params_input['rbin'] = r

                params_list.append(setup_input(params_input))
               
    exit()
    return params_list

def setup_input(params_input, looparams):

    params = {}


    """ === molecule directory ==== """ 
    if params_input['jobtype'] == "slurm":
        params['main_dir']      = "/gpfs/cfel/cmi/scratch/user/zakemil/PECD/pecd/" 
        params['working_dir']   = "/gpfs/cfel/cmi/scratch/user/zakemil/PECD/tests/molecules/c/"
    elif params_input['jobtype'] == "local":
        params['main_dir']      = "/Users/zakemil/Nextcloud/projects/PECD/pecd/"#
        params['working_dir']   = "/Users/zakemil/Nextcloud/projects/PECD/tests/molecules/c/"


    params['rot_wf_file']       = params['working_dir'] + "rv_wavepackets/" + "wavepacket_J60.h5"
    params['rot_coeffs_file']   = params['working_dir'] + "rv_wavepackets/" + "coefficients_j0_j60.rchm"


    """==== file paths and names ===="""

    params['file_psi0']         =   "psi0_" + params['molec_name']   + \
                                    "_" + str(params['bound_nbins'])   + \
                                    "_" + str(params['bound_nlobs']) + \
                                    "_" + str(params['bound_binw'])    + \
                                    "_" + str(params['bound_lmax'])  + \
                                    "_" + str(params['esp_method_name'])   + ".dat"

    params['file_hmat0']        =   "hmat0_" + params['molec_name']   + \
                                    "_" + str(params['bound_nbins'])   + \
                                    "_" + str(params['bound_nlobs']) + \
                                    "_" + str(params['bound_binw'])    + \
                                    "_" + str(params['bound_lmax'])  + \
                                    "_" + str(params['esp_method_name'])   + ".dat"

    params['file_enr0']         =   "enr0_" + params['molec_name']   + \
                                    "_" + str(params['bound_nbins'])   + \
                                    "_" + str(params['bound_nlobs']) + \
                                    "_" + str(params['bound_binw'])    + \
                                    "_" + str(params['bound_lmax'])  + \
                                    "_" + str(params['esp_method_name'])   + ".dat"

    params['file_quad_levels']  =   "quad_levels_" + params['molec_name']   + \
                                    "_" + str(params['bound_nbins'])   + \
                                    "_" + str(params['bound_nlobs']) + \
                                    "_" + str(params['bound_binw'])    + \
                                    "_" + str(params['bound_lmax'])  + \
                                    "_" + str(params['esp_method_name'])  + \
                                    "_" + str(params['sph_quad_tol'])   + ".dat"

    params['file_esp']          =   "esp_" + params['molec_name']   + \
                                    "_" + str(params['bound_nbins'])   + \
                                    "_" + str(params['bound_nlobs'])   + \
                                    "_" + str(params['bound_binw'])    + \
                                    "_" + str(params['bound_lmax'])    + \
                                    "_" + str(params['esp_method_name'])    + ".dat"
    

    params['job_directory'] =  params['working_dir'] + params['molec_name']   + \
                                "_" + str(params['bound_nbins'])   + \
                                "_" + str(params['bound_nlobs'])   + \
                                "_" + str(params['bound_binw'])    + \
                                "_" + str(params['bound_lmax'])    + \
                                "_" + str(params['esp_method_name']) +"/"




    #params['euler0'] = [0.0, np.pi/4.0, 0.0] #alpha, beta, gamma [radians]

    # generate 3D grid of Euler angles
    #params['n_euler']   = 2 # number of points per Euler angle. Total number of points is n_euler**3
    

    params['nlobs']     = params['bound_nlobs']
    params['nbins']     = 0
    params['binw']      = params['bound_binw']

    params['FEMLIST']   = [     [params['bound_nbins'], params['bound_nlobs'], params['bound_binw']] ,\
                                [params['nbins'], params['nlobs'], params['binw']] ] 



    params['plot_ini_orb']      = False #plot initial orbitals? iorb = 0,1, ..., ivec + 1
    params['calc_free_energy']  = False #calculate instantaneous energy of the free electron wavepacket in the field



    time_to_au                   = CONSTANTS.time_to_au[ params['time_units'] ]

    params['save_ham_init']      = True #save initial hamiltonian in a file for later use?
    params['save_psi_init']      = True
    params['save_enr_init']      = True

    
    params['plot_elfield']       = False

    params['wavepacket_file']    = "wavepacket"

    params['file_hmat_init']      =   "hmat_init_" + params['molec_name']   + \
                                    "_" + str(params['bound_nbins'] + params['nbins'])   + \
                                    "_" + str(params['bound_nlobs'] + params['nbins'] * params['nlobs']) + \
                                    "_" + str(params['bound_binw'])    + \
                                    "_" + str(params['bound_lmax'])  + \
                                    "_" + str(params['esp_method_name'])  

    params['file_psi_init']       =   "psi_init_" + params['molec_name']   + \
                                    "_" + str(params['bound_nbins'] + params['nbins'])   + \
                                    "_" + str(params['bound_nlobs'] + params['nbins'] * params['nlobs']) + \
                                    "_" + str(params['bound_binw'])    + \
                                    "_" + str(params['bound_lmax'])  + \
                                    "_" + str(params['esp_method_name'])  

    params['file_enr_init']       =   "enr_init_" + params['molec_name']   + \
                                    "_" + str(params['bound_nbins'] + params['nbins'])   + \
                                    "_" + str(params['bound_nlobs'] + params['nbins'] * params['nlobs']) + \
                                    "_" + str(params['bound_binw'])    + \
                                    "_" + str(params['bound_lmax'])  + \
                                    "_" + str(params['esp_method_name']) 




    if freq_units == "nm":
        params['omega']     = 10**9 *  CONSTANTS.vellgt / params['omega'] # from wavelength (nm) to frequency  (Hz)
    elif freq_units == "ev":
        params['omega']     = CONSTANTS.ev_to_hz * params['omega']   # from ev to frequency  (Hz)
    else:
        raise ValueError("Incorrect units for frequency")

    opt_cycle                  = 1.0e18/params['omega']
    suggested_no_pts_per_cycle = 50     # time-step can be estimated based on the carrier frequency of the pulse. Guan et al. use 1000 time-steps per optical cycle (in small Krylov basis). We can use much less. Demekhin used 50pts/cycle
    # 1050 nm = 1.179 eV = 285 THz -> 1 optical cycle = 3.5 fs
    
    print( "Electric field carrier frequency = " + str("%10.3f"%( params['omega'] * 1.0e-12 )) + " THz" )
    print( "Electric field oscillation period (optical cycle) = " + str("%10.3f"%(1.0e15/params['omega'])) + " fs")
    print( "suggested time-step for field linear frequency = " + str("%12.3f"%(params['omega']/1e12)) + " THz is: " + str("%10.2f"%(opt_cycle/suggested_no_pts_per_cycle )) + " as")

    params['omega'] *= 2.0 * np.pi # linear to angular frequency
    params['omega'] *= CONSTANTS.freq_to_au['Hz'] #Hz to a.u.

    """ ---- field intensity ----- """
    
    #params['E0'] = 1.0e9 #V/cm
    
    #field strength in a.u. (1a.u. = 5.1422e9 V/cm). For instance: 5e8 V/cm = 3.3e14 W/cm^2
    #convert from W/cm^2 to V/cm


    field_units     = "V/cm" 
    field_strength  = np.sqrt(intensity/(CONSTANTS.vellgt * CONSTANTS.epsilon0))
    print("field strength = " + "  %8.2e"%field_strength)

    params['E0']        = field_strength
    params['E0']        *= CONSTANTS.field_to_au[field_units] 





    """==== field dictionaries ===="""

    field_RCPL   = {"omega":            params['omega'], 
                    "E0":               params['E0'], 
                    "CEP0":             0.0, 
                    "spherical":        True}
                    
    field_LCPL   = {"omega":            params['omega'], 
                    "E0":               params['E0'], 
                    "CEP0":             0.0, 
                    "spherical":        True}

    field_LP    = { "omega":            params['omega'], 
                    "E0":               params['E0'], 
                    "CEP0":             0.0}


    # if gaussian width is given: e^-t^2/sigma^2
    # FWHM = 2.355 * sigma/sqrt(2)

    params['opt_cycle'] = 2.0 * np.pi /params['omega'] 

    env_gaussian = {"function_name": "envgaussian", 
                    "FWHM": 2.355 * (time_to_au * params['tau'])/np.sqrt(2.0), 
                    "t0": (time_to_au * params['tc'])  }

    env_sin2 = {"function_name": "envsin2", 
                    "Ncycles": 10 , 
                    "t0": (time_to_au * params['tc']),
                    "t_cycle": params['opt_cycle']  }




if __name__ == "__main__":    


    inputfile 	= "input_c" #input file name

	import importlib
	input_module = importlib.import_module(inputfile)

	print("jobtype: " + str(jobtype))
	print("input file: " + str(inputfile))
	
    params_input = input_module.read_input()


	params_list = gen_inputs_list(jobtype,params_input)




	jobdir = create_dirs(params)
	run_propagate(N_euler,N_batches,jobtype,inputfile,jobdir)