import numpy as np
import os
import subprocess

def create_dirs(params,N_euler_3D):

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



def run_propagate(N_euler,N_batches,jobtype,inputfile,jobdir):

	if jobtype == "maxwell":
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

    lmin = 
    lmax = 

    nlobmin = 
    nlobmax = 

    rbinmin = 
    rbinmax = 

    for 

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


    """ === molecule definition ==== """ 
    params['mol_geometry']  = {"rc":0.0} #angstroms
    params['mol_masses']    = {"c":12.0}
    params['mol_embedding'] = "bisector" #TROVE's bisector embedding
    params['N_euler'] 	    = 1 #number of euler grid points per dimension
    params['N_batches'] 	= 1 #number of batches for orientation averaging


    """ === ro-vibrational part ==== """ 
    params['density_averaging'] = False #use rotational proability density for orientation averageing. Otherwise uniform probability. 
    params['rot_wf_file']       = params['working_dir'] + "rv_wavepackets/" + "wavepacket_J60.h5"
    params['rot_coeffs_file']   = params['working_dir'] + "rv_wavepackets/" + "coefficients_j0_j60.rchm"
    params['Jmax']              = 60 #maximum J for the ro-vibrational wavefunction
    params['rv_wavepacket_time']= 50
    params['rv_wavepacket_dt']  = 0.1 #richmol time-step in ps #


    """==== BOUND ===="""
    params['map_type']      = 'DVR' #DVR or SPECT
    params['hmat_format']   = "sparse_csr" # numpy_arr
    params['file_format']   = 'dat' #npz, hdf5

    """==== basis set parameters for BOUND ===="""

    params['bound_nlobs']   = 10
    params['bound_nbins']   = 50
    params['bound_binw']    = 1.65
    params['bound_rshift']  = 0.0
    params['bound_lmax']    = 2
    
    params['save_ham0']     = True #save the calculated bound state Hamiltonian
    params['save_psi0']     = True #save psi0
    params['save_enr0']     = True #save eigenenergies for psi0

    params['num_ini_vec']   = 20 # number of initial wavefunctions (orbitals) stored in file

    """ ARPACK eigensolver parameters """
    params['ARPACK_tol']        = 1e-3
    params['ARPACK_maxiter']    = 60000
    params['ARPACK_enr_guess']  = None # (eV)
    params['ARPACK_which']      = 'LA'
    params['ARPACK_mode']       = "normal"

    """==== potential energy matrix ===="""
    params['read_ham_init_file'] = False #if available read the prestored initial hamiltonian from file
    params['gen_adaptive_quads'] = True
    params['use_adaptive_quads'] = True
    params['sph_quad_global']    = "lebedev_023" #global quadrature scheme in case we don't use adaptive quadratures.
    params['sph_quad_tol']       = 1e-5
    params['calc_method']        = 'jit' #jit, quadpy, vec
    params['hmat_filter']        = 1e-4 #threshold value for keeping matrix elements of field-free Ham

    """==== electrostatic potential ===="""
    params['integrate_esp'] = False
    params['esp_method_name']    = "uhf_631Gss"
    params['esp_mode']           = "exact" #exact or interpolate
    params['enable_cutoff']      = True #use cut-off for the ESP?
    params['r_cutoff']           = 40.0    
    params['plot_esp']           = False

    params['scf_enr_conv']       = 1.0e-6 #convergence threshold for SCF
    params['scf_basis']          = '6-31G**' #"cc-pDTZ" #"631G**"
    params['scf_method']         = 'uhf'

    params['esp_rotation_mode']  = 'mol_xyz' #'on_the_fly', 'to_wf'

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


    """==== PROPAGATE ===="""

    #params['euler0'] = [0.0, np.pi/4.0, 0.0] #alpha, beta, gamma [radians]

    # generate 3D grid of Euler angles
    #params['n_euler']   = 2 # number of points per Euler angle. Total number of points is n_euler**3
    

    params['nlobs']     = params['bound_nlobs']
    params['nbins']     = 0
    params['binw']      = params['bound_binw']

    params['FEMLIST']   = [     [params['bound_nbins'], params['bound_nlobs'], params['bound_binw']] ,\
                                [params['nbins'], params['nlobs'], params['binw']] ] 


    params['t0']        = 0.0 
    params['tmax']      = 4.0 
    params['dt']        = 1.5
    params['ivec']      = 0 

    params['plot_ini_orb']      = False #plot initial orbitals? iorb = 0,1, ..., ivec + 1
    params['calc_free_energy']  = False #calculate instantaneous energy of the free electron wavepacket in the field



    params['time_units']         = "as"
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


    """ ====== FIELD PARAMETERS ====== """

    """ ---- carrier frequency ----- """
    params['omega']     = 53.6057 #23.128 = 54 eV, 60nm = 20 eV
    freq_units          = "ev" #nm or ev

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
    field_units     = "V/cm"
    #field strength in a.u. (1a.u. = 5.1422e9 V/cm). For instance: 5e8 V/cm = 3.3e14 W/cm^2
    #convert from W/cm^2 to V/cm

    intensity       = 3.0e+16 #W/cm^2 #peak intensity

    field_strength  = np.sqrt(intensity/(CONSTANTS.vellgt * CONSTANTS.epsilon0))
    print("field strength = " + "  %8.2e"%field_strength)

    params['E0']        = field_strength
    params['E0']        *= CONSTANTS.field_to_au[field_units] 

    """ ---- field params----- """
    params['tau']       = 1000.0 #as: pulse duration (sigma)
    params['tc']        = 2000.0 #as: pulse centre
    

    """==== field dictionaries ===="""

    field_CPL   = { "function_name":    "fieldCPL", 
                    "omega":            params['omega'], 
                    "E0":               params['E0'], 
                    "CEP0":             0.0, 
                    "spherical":        True, 
                    "typef":            "LCPL"}

    field_LP    = { "function_name":    "fieldLP", 
                    "omega":            params['omega'], 
                    "E0":               params['E0'], 
                    "CEP0":             0.0}



    params['field_form'] = "analytic" #or numerical
    params['field_type'] = field_LP 

    """ Available field types :
        1) field_CPL
        2) field_LP
        3) field_omega2omega
    """

    # if gaussian width is given: e^-t^2/sigma^2
    # FWHM = 2.355 * sigma/sqrt(2)
    env_gaussian = {"function_name": "envgaussian", 
                    "FWHM": 2.355 * (time_to_au * params['tau'])/np.sqrt(2.0), 
                    "t0": (time_to_au * params['tc'])  }

    params['opt_cycle'] = 2.0 * np.pi /params['omega'] 

    env_sin2 = {"function_name": "envsin2", 
                    "Ncycles": 10 , 
                    "t0": (time_to_au * params['tc']),
                    "t_cycle": params['opt_cycle']  }


    params['field_env'] = env_gaussian 

    """ Available envelopes :
        1) env_gaussian
        2) env_flat
    """

    """==== POST-PROCESSING: PLOTS ===="""

    params['plot_modes']    = { "snapshot":         True, 
                                "animation":        False}

    params['plot_types']    = { "radial":           False,
                                "angular":          False,
                                "r-radial_angular": True, 
                                "k-radial_angular": False} 

    params['plot_controls'] = { "plottimes":        list(np.linspace(0.0,params['tmax'],3)),#list(np.linspace(0.0,params['tmax'],150)),#200.0,300.0,600.0,700.0,800.0,900.0,1000.0],
                                "save_snapshots":   True,
                                "save_anim":        False,
                                "show_snapshot":    False,
                                "show_anim":        False, 
                                "fname_snapshot":   "obs",
                                "fname_animation":  "anim_obs"}

    """ plotrate : rate of plotting observables in timestep units in animated plots
        plottimes: times (in time_units) at which we plot selected observables in a static graph
        save_static: save single shot plots to appropriate files (named separately for each plottime)
        save_anim: save animation in a file
        show_static: show single shot plots during the analysis
        show_anim: show animation at the end of analysis
        static_filename: name of the file into which the snapshots will be saved
        animation_filename: name of the file into which animations will be saved
    """

    params["save_snapthots"] = True


    """==== momentum-space distributions ===="""
    """ PECD """
    params['analyze_pecd']    = False
    params['pecd_lmax']       = 2 #maximum angular momentum in the spherical harmonics expansion of the momentum probability function
    params['k_pecd']          = [0.3,0.47,0.7,0.9] #(a.u.) (list) at what electron momentum do you want PECD?
    params['analyze_time']    = params['tmax']  #at what time(s) (in as) do we want to calculate PECD and other observables?
    
    """ MPADs """
    params['analyze_mpad']    = True
    params['FT_method']       = "FFT_hankel" #"FFT_cart" #or quadratures
    params['N_r_points']      = 500 #number of radial points at which Hankel Transform is evaluated.
    # [15.0,50.0]
    params['k_list_pad']      =  list(np.linspace(1,2.0,4)) #list of wavevectors for MFPAD plots
    
    params['n_pes_pts']         = 1000 #numer of points for PES evaluation
    params['max_pes_en']        = 3.0 #in a.u.



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