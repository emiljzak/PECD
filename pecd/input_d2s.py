import numpy as np
import CONSTANTS
import os
import itertools
import GRAPHICS

def read_input():

    """ Set up essential input parameters"""

    params = {}

    """ ===== mode of execution ===== """ 
    """
        1) 'propagate':    propagate wavefunction for a grid of Euler angles and a grid of parameters
        2) 'analyze':      analyze wavefunction for a grid of Euler angles and a grid of parameters
    """
    
    params['mode']      = 'propagate'
    """
        In analyze mode the user specifies only basis set parameters and parameters in the 'analysis' section below
        All other parameters are read from respective input files.
    """
    
    """ ===== type of job ===== """ 
    """
        1) 'local':    local job 
        2) 'slurm':    submission to a SLURM workload manager for an HPC job
    """

    params['jobtype'] 	= "slurm" 



    params['job_label']    = "conv" #job identifier. In case of Psi4 ESP it can be metod/basis specification: "UHF-aug-cc-pVTZ" #"UHF_6-31Gss"


    """====== Basis set parameters for BOUND ======"""
    """ 
        Set up basis set parameters for the calculation of the stationary Hamiltonian matrix. 
        Bound states are calculated with these parameters.
        Format (tuple): params['bound_nnn'] = (par_min, par_max, number_of_params) - to set up loop over parameters
    """
    """ BOUND PART"""
    params['bound_nlobs_arr']   = (10,10,1)
    params['bound_lmax_arr']    = (6,6,4)
    params['bound_binw_arr']    = (1.0,3.0,11)

    params['bound_nbins']   = 30
    params['bound_rshift']  = 0.0

    """ CONTINUUM PART"""

    params['map_type']      = 'DVR' #DVR, SPECT (mapping of basis set indices)

    """==== time-grid parameters ===="""

    params['time_units']    = "as"

    params['t0']            = 0.0 
    params['tmax']          = 4.0 
    params['dt']            = 2.0 # replace with the calculated number for N points per cycle
    params['wfn_saverate']  = 1 #save rate wrt. index labeling the timegrid. '1' means save all time-steps



    """==== Molecule-field orientation ===="""

    params['N_euler'] 	        = 1     # number of euler grid points per dimension (beta angle) for orientation averaging. Alpha and gamma are on double-sized grid.
    params['N_batches'] 	    = 1    # number of batches for orientation averaging
    params['orient_grid_type']  = "2D"  # 2D or 3D. Use 2D when averaging is performed over phi in W2D.

    """ ===== Molecule definition ====== """ 
    """
        Set up properties of the molecule, including atomic masses, geometry, MF embedding
        *)  mol_geometry: for now only supports a dictionary with internal coordinates (bond lengths, angles). 
            Extend later to the option of the cartesian input. 
        *) mol_embedding (string): define the MF embedding, which matches the embedding used for calculating the ro-vibrational wavefunctions.
            Extend later beyond the TROVE wavefunctions. Add own ro-vibrational wavefunctions and embeddings.
    """

    params['molec_name']    = "d2s"
    params['mol_geometry']  = {"rSD1":1.336, "rSD2": 1.2 * 1.336, "alphaDSD": 92.0} #angstroms#in next generation load internal geometry from file
    params['mol_masses']    = {"S":32.0, "D":2.0}
    params['mol_embedding'] = "bisector" #TROVE's bisector embedding

    params['sph_quad_tol']       = 1e-10     # tolerance (in a.u.) for the convergence of matrix elements


    """ __________________________ PROPAGATE BLOCK __________________________"""

    if params['mode'] == "propagate":


        """ ====== Initial wavefunction ====="""

        params['ivec']          = 8 #ID of eigenstate to propagate
                                #Later extend to arbitrary linear combination of eigenvector or basis set vectors.


        """ ====== FIELD PARAMETERS ====== """

        params['freq_units']    = "ev"      # nm or ev
        params['omega']         = 40.0   # 23.128 nm = 54 eV, 60 nm = 20 eV
        params['intensity']     = 1.0e+14   # W/cm^2: peak intensity

        """ Available field types :
            1) RCPL   - right-circularly polarized field
            2) LCPL    - left-circularly polarized field
            3) LP      - linearly polarized field
        """

        """ Available envelopes :
            1) gaussian
            2) sin2 
        """
        
        params['field_form']    = "analytic" #or numerical (i.e. read from file). To be implemented.

        params['field_func_name']    = "RCPL"
        params['field_env_name']     = "gaussian" 

        """ gaussian pulse """
        params['gauss_tau']     = 1000.0 #as: pulse duration (sigma)
        params['gauss_t0']      = 2000.0 #as: pulse centre

        """ sin2 pulse """
        params['sin2_ncycles']  = 10
        params['sin2_t0']       = 2000.0

        params['CEP0']          = 0.0 #CEP phase of the field


        """===== Potential energy matrix ====="""
        
      
        params['gen_adaptive_quads'] = False # generate adaptive quadratures and save their parameters in a file?

        params['use_adaptive_quads'] = False          # read adaptive quadrature parameters from file and use them
        params['sph_quad_default']   = "lebedev_131" # global quadrature scheme in case we do not use adaptive quadratures.

        params['calc_method']        = 'jit' #jit, quadpy, vec: use jit, quadpy or vector implementation of the matrix elements

        """ **** parameters of the multipole moment expansion of the ESP **** """
        params['multi_lmax']         = 8 #maximum l in the multipole expansion
        params['multi_ncube_points'] = 201
        params['multi_box_edge']     = 20

        """==== electrostatic potential ===="""


        params['esp_mode']           = "exact" #exact or multipoles or anton. 
                                        # exact -> use Psi4. 
                                        # multipoles -> perform multipole expansion of the potential from given charge distr.
                                        # anton -> partial wave representation of the potential from A. Artemyev
                                        # use anton with nlobs = 10, nbins = 200, Rbin = 2.0, lmax = 9, Lmax = 8. 1800 grid points. 160k basis size.

        params['enable_cutoff']      = True #use cut-off for the ESP?
        #params['r_cutoff']           = 40.0    

        params['scf_enr_conv']       = 1.0e-6 #convergence threshold for SCF
        params['scf_basis']          = 'aug-cc-pVTZ' #"cc-pDTZ" #"631G**"
        params['scf_method']         = 'UHF'

        params['esp_rotation_mode']  = 'mol_xyz' #'on_the_fly', 'to_wf'
        params['plot_esp']           = False
        params['integrate_esp']      = False #integrate ESP?


        params['calc_free_energy']  = False #calculate instantaneous energy of the free electron wavepacket in the field


        """===== Hamiltonian parameters ====="""
        params['read_ham_init_file']    = False    # if available read the initial Hamiltonian from file
        params['hmat_format']           = "sparse_csr" # numpy_arr
        params['hmat_filter']           = 1e-12 #threshold value (in a.u.) for keeping matrix elements of the field-free Hamiltonian

        params['num_ini_vec']           = 20 # number of initial wavefunctions (orbitals) stored in file
        params['file_format']           = 'npz' #dat, npz, hdf5 (format for storage of the wavefunction and the Hamiltonian matrix)

        #params['']

        """ ===== ARPACK eigensolver parameters ===== """

        params['ARPACK_tol']        = 1e-8      # error tolerance (relative)
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

        params['save_ham_init']  = True #save initial hamiltonian in a file for later use?
        params['save_psi_init']  = True
        params['save_enr_init']  = True

        params['wavepacket_format'] = "h5" #dat or h5

        params['plot_elfield']      = False
        params['plot_ini_orb']      = False #plot initial orbitals? iorb = 0,1, ..., ivec + 1

    


    elif params['mode'] == "analyze":
        """ __________________________ ANALYZE BLOCK __________________________"""

        """
                Available functions:

                    1) rho1Dr:  rho(r,theta0,t)     - radial density for fixed angle
                    2) rho1Dth: rho(r0,theta,t)     - angular density for fixed distance
                    3) rho2D:   rho(r,theta,phi0,t) - polar 2D map, fixed phi0
                    4) rho2Dav: rho_av(r,theta,t)   - polar 2D map, averaged over phi
                    5) rho3Dcart: rho3D(x,y,z,t)    - 4D plot with Mayavi


                    1) W1Dth:  W(k0,theta,t)     - 1D momentum probability density in theta 
                    2) W1Dk: W(k,theta0,t)      -   
                    3) W2D:  W(k,theta,phi0,t)     - 2D momentum probability density in k,theta
                    4) W2Dav: W(k,theta,t)     - 2D momentum probability density in k,theta, phi-averaged
                    5) PECD
        """
        

        
        rho2D = {   'name':         'rho2D',
                    'plane':        ('XY','XZ','YZ'), #in which Cartesian planes do we want to plot rho2D? 'XY','XZ','YZ' or [nx,ny,nz] - vector normal to the plane
                    'plot':         (True, GRAPHICS.gparams_rho2D_polar()), #specify parameters of the plot to load
                    'show':         True, # show image on screen                    
                    'save':         True,
                    'r_grid':       {   'type':'manual', #manual or automatic grid type. 
                                        'npts': 100,    #ignored when automatic (2*rmax)
                                        'rmin': 0.0,    #ignored when automatic
                                        'rmax': 30.0  #ignored when automatic
                                        #Automatic means that we choose ranges based on maximum range given by the basis set.   
                                    },                   
                    'th_grid':      (0.0,2.0*np.pi,360),
                    'coeff_thr':    1e-6 #threshold for the wavefunction coefficients in the calculation of rho
                    }

        W2D = {     'name':         'W2D',
                    'plane':        ('YZ',), #in which Cartesian planes do we want to plot rho2D? 'XY','XZ','YZ' or [nx,ny,nz] - vector normal to the plane
                    'plot':         (True, GRAPHICS.gparams_W2D_polar()), #specify parameters of the plot to load
                    'show':         False, # show image on screen
                    'save':         False, # save array in file
                                    # Momentum grid parameters only for plotting purposes
                    'k_grid':       {   'type':'manual', #manual or automatic grid type. 
                                        'npts': 500,    #ignored when automatic (2*rmax)
                                        'kmin': 0.0,    #ignored when automatic
                                        'kmax': 2.0  #ignored when automatic
                                        #Automatic means that we choose ranges based on maximum range given by the basis set.   
                                    },                   
                    'th_grid':      (0.0,2.0*np.pi,360),
                    'legendre':     False, # calculate Legendre decomposition

                    'PES':          True, # calculate PES
                    'PES_params': {     
                                    'name':         'PES',
                                    'plot':         (True, GRAPHICS.gparams_PES()), #specify parameters of the plot to load
                                    'show':         True, # show image on screen
                                    'save':         True, # save array in file
                                
                                    'k-axis':       "energy", # energy (eV) or momentum (a.u.)
                                    'y-axis':       "log",   # log or unit scale
                                    'normalize':    True,   # normalize the cross-section
                                    },

                    }


        W2Dav = {   'name':         'W2Dav',
                    'plot':         (True, GRAPHICS.gparams_W2Dav_polar()), #specify parameters of the plot to load
                    'show':         True, # show image on screen
                    'save':         False, # save array in file
                                    # Momentum grid parameters only for plotting purposes
                    'k_grid':       {   'type':'automatic', #manual or automatic grid type. 
                                        'npts': 500,    #ignored when automatic (2*rmax)
                                        'kmin': 0.0,    #ignored when automatic
                                        'kmax': 2.0  #ignored when automatic
                                        #Automatic means that we choose ranges based on maximum range given by the basis set.   
                                    },                   
                    'th_grid':      (0.0,2.0*np.pi,360),
                    
                    'nphi_pts':     2, #number of phi points for the integration over tha azimuthal angle.
                    
                    'legendre':     True, # calculate Legendre decomposition



                    'PES':          False, # calculate PES
                    'PES_params': {     
                                    'name':         'PES',
                                    'plot':         (True, GRAPHICS.gparams_PES()), #specify parameters of the plot to load
                                    'show':         True, # show image on screen
                                    'save':         True, # save array in file
                                
                                    'k-axis':       "energy", # energy (eV) or momentum (a.u.)
                                    'y-axis':       "log",   # log or unit scale
                                    'normalize':    True,   # normalize the cross-section
                                    },

                    }

        PECD = {    'name':         'PECD',
                    'plot2D':       (True, GRAPHICS.gparams_PECD2D()), #specify parameters of the plot to load
                    'plot1D':       (True, GRAPHICS.gparams_PECD1D()), #specify parameters of the plot to load
                    'orient_av':    False, # perform orientation averaging over the Euler angle's grid?
                    'show':         True, # show images on screen
                    'save':         True, # save arrays in files
                    'kmax':         2.0,
                    
                    }

        bcoeffs = { 'name':     'bcoeffs',
                    'plot':     (True, GRAPHICS.gparams_barray2D()),
                    }

        #params['obs_params_rho'] = rho2D
        #params['obs_params_W2D'] = W2D
        #params['obs_params_W2Dav'] = W2Dav
        #params['obs_params_PECD'] = PECD


        params['space_analyze_times']    =   list(np.linspace(0.0, params['tmax'], 1 ))
        params['momentum_analyze_times'] =   list(np.linspace(params['tmax'], params['tmax'], 1 ))

        params['analyze_space']     = [rho2D]
        params['analyze_momentum']  = []
        

        params['PECD']  = PECD
        params['W2Dav'] = W2Dav
        params['W2D']   = W2D
        params['rho2D'] = rho2D
        params['bcoeffs'] = bcoeffs


        """ *** Momentum-space wavefunction *** """
        params['FT_method']       = "FFT_hankel"    # "FFT_cart" #or quadratures
        # Fourier transform is calculated from the wavefunction calculated on real-space grid bounded by rcutoff and Rmax.
        params['npts_r_ft']       = 500             # number of radial points over which the Hankel Transform is evaluated.
        params['rcutoff']         = 20.0            # radial cut-off of the wavepacket in the calculation of momentum space distributions
       
        params['plot_Plm']        = False           # plot and save photoelectron partial waves?
        params['plot_Flm']        = False           # plot and save individual Hankel transforms?

        """ *** Legendre expansion *** """
        params['Leg_lmax']          = 6      # maximum angular momentum in the Legendre expansion
        params['Leg_plot_reconst']  = True   # plot and compare the reconstructed distribution
        params['Leg_test_interp']   = False  # test interpolation of W2D by plotting
        params['plot_bcoeffs']      = True  # plot b-coefficients
        params['Leg_npts_r']        = 500   # number of radial points for plotting of the Legendre expansion
        params['Leg_npts_th']       = 360   # number of angular points for plotting of the Legendre expansion
            
        """ *** PES *** """
        params['pes_npts']       = 1000    # numer of points for PES evaluation
        params['pes_max_k']      = 2.0     # maximum momentum in a.u. Must be lower than the momentum range for W2D
        params['pes_lmax']       = 1

        """ *** PECD *** """
        params['pecd_lmax']       = 6               # maximum angular momentum in the spherical harmonics expansion of the momentum probability function
        params['pecd_momenta']    = [1.35]   # (a.u.) (list) at what values of the electron momentum do you want PECD?


    return params
