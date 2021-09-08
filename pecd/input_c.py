import numpy as np
import CONSTANTS
import os
import itertools
def gen_input(jobtype):
    """ Set up essential input parameters"""

    params = {}

    """ ===== mode of execution ===== """ 
    """
        1) 'propagate':    propagate wavefunction for a grid of Euler angles and a grid of parameters
        2) 'analyze':      analyze wavefunction for a grid of Euler angles and a grid of parameters
    """
    
    params['mode']      = 'propagate'
    
    
    """ ===== type of job ===== """ 
    """
        1) 'local':    local job 
        2) 'slurm':    submission to a SLURM workload manager for an HPC job
    """

    params['jobtype'] 	= "local" #maxwell


    """ ===== Molecule definition ====== """ 
    """
        Set up properties of the molecule, including atomic masses, geometry, MF embedding
        *)  mol_geometry: for now only supports a dictionary with internal coordinates (bond lengths, angles). 
            Extend later to the option of the cartesian input. 
        *) mol_embedding (string): define the MF embedding, which matches the embedding used for calculating the ro-vibrational wavefunctions.
            Extend later beyond the TROVE wavefunctions. Add own ro-vibrational wavefunctions and embeddings.
    """

    params['molec_name']    = "c"
    params['mol_geometry']  = {"rc":0.0} #angstroms
    params['mol_masses']    = {"c":12.0}
    params['mol_embedding'] = "bisector" #TROVE's bisector embedding


    """====== Basis set parameters for BOUND ======"""
    """ 
        Set up basis set parameters for the calculation of the stationary Hamiltonian matrix. 
        Bound states are calculated with these parameters.
        Format (tuple): params['bound_nnn'] = (par_min, par_max, number_of_params) - to set up loop over parameters
    """
    params['bound_nlobs']   = (10,10,1)
    params['bound_lmax']    = (2,2,1)
    params['bound_binw']    = (1.4,1.4,1)

    params['bound_nbins']   = 20
    params['bound_rshift']  = 0.0

    params['N_euler'] 	    = 1 #number of euler grid points per dimension for orientation averaging
    params['N_batches'] 	= 1 #number of batches for orientation averaging

    params['map_type']      = 'DVR' #DVR, SPECT (mapping of basis set indices)


    """==== Propagation parameters ===="""

    params['ivec']          = 0 #ID of eigenstate to propagate
                            #Later extend to arbitrary linear combination of eigenvector or basis set vectors.

    params['time_units']    = "as"

    params['t0']            = 0.0 
    params['tmax']          = 4.0 
    params['dt']            = 1.5



    """ ====== FIELD PARAMETERS ====== """

    params['freq_units']    = "ev"      # nm or ev
    params['omega']         = 53.6057   # 23.128 nm = 54 eV, 60 nm = 20 eV
    params['intensity']     = 3.0e+16   # W/cm^2: peak intensity

    """ Available field types :
        1) field_RCPL    - right-circularly polarized field
        2) field_LCPL    - left-circularly polarized field
        2) field_LP      - linearly polarized field
        3) field_omega2omega
    """

    """ Available envelopes :
        1) env_gaussian
        2) env_sin2 
        2) env_flat
    """
    
    params['field_form']    = "analytic" #or numerical (i.e. read from file). To be implemented.

    params['field_type']    = field_LP
    params['field_env']     = env_gaussian

    params['gauss_tau']       = 1000.0 #as: pulse duration (sigma)
    params['gauss_tc']        = 2000.0 #as: pulse centre
    

    """===== Potential energy matrix ====="""
    
    params['read_ham_init_file'] = False    # if available read the initial Hamiltonian from file
    params['gen_adaptive_quads'] = True     # generate adaptive quadratures and save their parameters in a file?
    params['sph_quad_tol']       = 1e-5     # tolerance (in a.u.) for the convergence of matrix elements

    params['use_adaptive_quads'] = True          # read adaptive quadrature parameters from file and use them
    params['sph_quad_default']   = "lebedev_023" # global quadrature scheme in case we do not use adaptive quadratures.

    params['calc_method']        = 'jit' #jit, quadpy, vec: use jit, quadpy or vector implementation of the matrix elements

    """==== electrostatic potential ===="""

    params['esp_method_name']    = "UHF_6-31Gss"
    params['esp_mode']           = "exact" #exact or interpolate
    params['enable_cutoff']      = True #use cut-off for the ESP?
    params['r_cutoff']           = 40.0    

    params['scf_enr_conv']       = 1.0e-6 #convergence threshold for SCF
    params['scf_basis']          = '6-31G**' #"cc-pDTZ" #"631G**"
    params['scf_method']         = 'UHF'

    params['esp_rotation_mode']  = 'mol_xyz' #'on_the_fly', 'to_wf'
    params['plot_esp']           = False
    params['integrate_esp']      = False #integrate ESP?

    """===== Hamiltonian parameters ====="""
    
    params['hmat_format']   = "sparse_csr" # numpy_arr
    params['hmat_filter']   = 1e-2 #threshold value (in a.u.) for keeping matrix elements of the field-free Hamiltonian

    params['num_ini_vec']   = 20 # number of initial wavefunctions (orbitals) stored in file
    params['file_format']   = 'dat' #dat, npz, hdf5 (format for storage of the wavefunction and the Hamiltonian matrix)


    """ ===== ARPACK eigensolver parameters ===== """

    params['ARPACK_tol']        = 1e-3      # error tolerance (relative)
    params['ARPACK_maxiter']    = 60000     # maximum number of iterations
    params['ARPACK_enr_guess']  = None      # energy guess for the shift inverse mode in (eV)
    params['ARPACK_which']      = 'LA'      # LA, SM, SA, LM
    params['ARPACK_mode']       = "normal"  # normal or inverse



    """ === ro-vibrational part ==== """ 
    params['density_averaging'] = False #use rotational proability density for orientation averageing. Otherwise uniform probability. 

    params['Jmax']              = 60 #maximum J for the ro-vibrational wavefunction
    params['rv_wavepacket_time']= 50
    params['rv_wavepacket_dt']  = 0.1 #richmol time-step in ps #

    """====  SAVING ===="""
    params['save_ham0']     = True #save the calculated bound state Hamiltonian
    params['save_psi0']     = True #save psi0
    params['save_enr0']     = True #save eigenenergies for psi0

    params["save_snapthots"] = True
    params['save_ham_init']      = True #save initial hamiltonian in a file for later use?
    params['save_psi_init']      = True
    params['save_enr_init']      = True



    """==== POST-PROCESSING: PLOTS ===="""
    params['plot_elfield']       = False
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




    return params
