import numpy as np
def gen_input():

    params = {}
    #move all constants to constants.py module!
    time_to_au =  np.float64(1.0/24.188)
    freq_to_au = np.float64(0.057/800.0)
    field_to_au =  np.float64(1.0/(5.14220652e+9))

    """ === molecule directory ==== """ 
    """ in this directory we read/write files associated with a given molecule """
    params['working_dir'] = "/Users/zakemil/Nextcloud/projects/PECD/tests/molecules/h2o/"
    params['molec_name'] = "h2o"

    """====basis set parameters===="""
    params['basis'] = "prim" # or adiab
    params['nlobatto'] = 5
    params['nbins'] = 4
    params['binwidth'] = 3.0
    params['rshift'] = 0.05 #rshift must be chosen such that it is non-zero and does not cover significant probability density region of any eigenfunction.
    params['lmin'] = 0
    params['lmax'] = 1
    
    """==== electrostatic potential ===="""
    params['pot_type'] = "grid" #type of potential: analytic or grid
    params['potential'] = "pot_hydrogen" # 1) pot_diagonal (for tests); 2) pot_hydrogen; 3) pot_null; 4) pot_grid_psi4_d2s
    params['potential_grid'] ="esp_grid_h2o_uhf_631Gss_10_0.5" #filename for grid representation of ESP. Only if pot_type = grid
    params['r_cutoff'] = 5.0 #cut-off radius for the cation electrostatic potential. We are limited by the capabilities of psi4, memory. Common sense says to cut-off the ESP at some range to avoid spurious operations
    """ Note: r_cutoff can be infered from quad_levels file: when matrix elements of esp are nearly an overlap between spherical funcitons, it is good r_in for setting esp=0"""

    """====runtime controls===="""
    params['method'] = "static" #static: solve field-free time-independent SE for a given potential, store matrix elements; dynamic_direct, dynamic_lanczos
    params['save_psi0_hamiltonian'] = True #save the calculated Hamiltonian used to get psi0 in a file?
    params['save_init_hamiltonian'] = True #save the calculated field-free Hamiltonian in a file?
    params['save_ini_wf'] = True #save initial wavefunction generated with eigenvec option to a file (spectral representation)
    params['read_ham_from_file'] = False #do we read stored initial hamiltonian matrix from file or generate it during run?
    params['gen_adiabatic_basis'] = True #generate eigenbasis of the initial Hamiltonian?


    params['ini_ham_file'] = "hmat_"+params['molec_name']+"_"+str(params['nbins'])+\
                        "_"+str(params['nlobatto'])+"_"+str(params['lmax'])+"_"+\
                            str(params['binwidth'])+"_"+params['potential_grid']+".dat"

   
    params['ini_energies_file'] = "energies0_"+params['molec_name']+"_"+str(params['nbins'])+\
                        "_"+str(params['nlobatto'])+"_"+str(params['lmax'])+"_"+\
                            str(params['binwidth'])+"_"+params['potential_grid']+".dat"



    params['t0'] = 0.0 
    params['tmax'] = 100.0 
    params['dt'] = 1.0 
    time_units = "as"
    params['sparse_format'] = False # True: store field-free matrices in csr format; False: store in full numpy array.

    """====initial state====""" 
    params['ini_state'] = "from_file" #from_file,spectral_manual, spectral_file, grid_1d_rad, grid_2d_sph,grid_3d,solve (solve static problem in Lobatto basis), eigenvec (eigenfunciton of static hamiltonian)
    params['ini_state_quad'] = ("Gauss-Laguerre",60) #quadrature type for projection of the initial wavefunction onto lobatto basis: Gauss-Laguerre, Gauss-Hermite
    params['ini_state_file_grid'] = "wf0grid.txt" #if requested: initial wavefunction on a 3D grid of (r,theta,phi)
    params['nbins_iniwf'] = 3 #number of bins in a reduced-size grid for generating the initial wavefunction by diagonalizing the static hamiltonian
    params['eigenvec_id'] = 0 #id (in ascending energy order) of the eigenvector of the static Hamiltonian to be used as the initial wavefunction for time-propagation. Beginning 0.
 
    params['n_ini_vec'] = 10 #number of initial wavefunctions (orbitals) stored


    params['ini_state_file'] = "psi0_"+params['molec_name']+"_"+str(params['nbins_iniwf'])+\
                        "_"+str(params['nlobatto'])+"_"+str(params['lmax'])+"_"+\
                            str(params['binwidth'])+"_"+params['potential_grid']+".dat"

    params['psi0_ham_file'] = "hmat_psi0_"+params['molec_name']+"_"+str(params['nbins_iniwf'])+\
                        "_"+str(params['nlobatto'])+"_"+str(params['lmax'])+"_"+\
                            str(params['binwidth'])+"_"+params['potential_grid']+".dat"


    """==== spherical quadratures ===="""
    params['scheme'] = "lebedev_011" #angular integration rule
    params['adaptive_quad'] =  True#True: read degrees of adaptive angular quadrature for each radial grid point. Use them in calculations of matrix elements.
    params['gen_adaptive_quads'] = False #generate and save in file a list of degrees for adaptive angular quadratures (Lebedev for now). This list is potential and basis dependent. To be read upon construction of the hamiltonian.
    params['quad_tol'] = 1e-3 #tolerance threshold for the convergence of potential energy matrix elements (global) using spherical quadratures
    params['atol'] = 1e-2 #absolute tolerance for the hamiltonian matrix to be non-symmetric
    params['rtol'] = 1e-2 #relative tolerance for the hamiltonian matrix to be non-symmetric
    params['quad_levels_file'] = "quad_levels_"+params['molec_name']+"_"+str(params['nbins'])+\
                        "_"+str(params['nlobatto'])+"_"+str(params['lmax'])+"_"+\
                            str(params['binwidth'])+"_"+ str('%5.2e'%params['quad_tol'])+\
                                "_"+params['potential_grid']+".dat"




    """===== TESTING ====="""
    params['test_potmat_accur'] = False #Test the accuracy of potential energy matrix elements 
    params['test_multipoles'] = False#test accuracy of potential energy matrix elements with multipole expansion of the potential and anlytic matrix elements
    params['test_lebedev'] = False #test accuracy of potential energy matrix elements with lebedev quadrature and exact potential
    params['multipoles_lmax'] = 6 #maximum L in the multipole expansion of the electrostatic potential
    params['plot_esp'] = True #plot ESP?
    params['calc_inst_energy'] = False #calculate instantaneous energy of a free-electron wavepacket?
    params['test_pads'] = False #test convergence etc. for momentum-space distributions and PECD


    """====molecule-field interaction hamiltonian===="""
    params['int_rep_type'] = 'spherical' #representation of the molecule-field interaction potential (spherical or cartesian ): for now only used in calculations of instantaneous electron wavepacket energy.
   

    """===== post-processing ====="""

    params['wavepacket_file'] = "wavepacket.dat" #filename into which the time-dependent wavepacket is saved
    params['plot_modes'] = {"single_shot": False, "animation": False}

    params['plot_types'] = { "radial": True,
                             "angular": True,
                             "r-radial_angular": True, 
                             "k-radial_angular": True} #decide which of the available observables you wish to plot

    params['plot_controls'] = { "plotrate": 1, 
                                "plottimes": [100.0],#200.0,300.0,600.0,700.0,800.0,900.0,1000.0],
                                "save_static": False,
                                "save_anim": False,
                                "show_static": True,
                                "show_anim": False, 
                                "static_filename": "obs",
                                "animation_filename": "anim_obs"}

    """ plotrate : rate of plotting observables in timestep units in animated plots
        plottimes: times (in time_units) at which we plot selected observables in a static graph
        save_static: save single shot plots to appropriate files (named separately for each plottime)
        save_anim: save animation in a file
        show_static: show single shot plots during the analysis
        show_anim: show animation at the end of analysis
        static_filename: name of the file into which the snapshots will be saved
        animation_filename: name of the file into which animations will be saved
    """

    """=== Fourier transform ==="""
    params['FT_method'] = "quadrature" #method for calculating the FT of the wavefunction: quadrature or fftn
    params['FT_output'] = "lebedev_grid" #interpolate_cart, interpolate_sph, lebedev_grid: how do you want your output FT?
    params['schemeFT_ang'] = "lebedev_025" #angular integration rule for calculating FT using the quadratures method
    params['schemeFT_rad'] = ("Gauss-Hermite",10) #quadrature type for projection of psi(t) onto the lobatto basis: Gauss-Laguerre, Gauss-Hermite
    params['schemeWLM'] = "lebedev_025" # 
    params['test_quadpy_direct'] = False #compare quadpy integration with manual lebedev. Test convergence of angular part of the FT
    params['test_conv_FT_ang'] = True #test spherical integration convergence in FT using quadratures

    params['pecd_lmax'] = 2 #maximum angular momentum of the expansion into spherical harmonics of the momentum probability function
    params['calculate_pecd'] = True #calculate FT of the wavefunction and expand it into spherical harmonics and calculate PECD?
    params['time_pecd'] = params['tmax'] #at what time (in a.u.) do we want to calculate PECD?
    params['WLM_quad_tol'] = 1e-4 #spherical quadrature convergence threshold for WLM(k_i)

    params['kmax'] = 0.5 #maximum value of wavevector in FT (a.u.). This value is set as primary and other quantities should be estimated based on it.
    #based on kmax and pulse duration we can estimate the spatial grid range:
    Rmax_est = 6.0 *  params['tmax'] * time_to_au *  params['kmax'] 
    print("An estimated range for spatial grid needed to cover " + str(params['kmax']) + " a.u. momentum wavepackets is: " + str("%10.0f"%(Rmax_est)) + " a.u.")
   
    params['nkpts'] = int( params['kmax'] *  params['nbins'] *  params['binwidth'] / (2.0 * np.pi) ) + 1
    print("Number of points per dimension in grid representation of the 3D-FT of the wavepacket = " + str("%5d"%params['nkpts'])) 

    #for prelimiary tests params['nkpts'] = 2
    params['nkpts'] = 1
    params['k-grid'] = np.linspace(0.5, params['kmax'],params['nkpts']) #range of momenta for the electron (effective radial range for the Fourier transform of the total wavefunction). Note that E = 1/2 * k^2, so it is easily convertible to photo-electron energy range
    

    """====field controls===="""
    params['plot_elfield'] = False #plot z-component of the electric field
    """ put most of what's below into a converion function """
    params['omega'] =  23.128 #nm or eV

    #convert nm to THz:
    vellgt     =  2.99792458E+8 # m/s
    params['omega']= 10**9 *  vellgt / params['omega'] # from wavelength (nm) to frequency  (Hz)
    opt_cycle = 1.0e18/params['omega']
    suggested_no_pts_per_cycle = 25     # time-step can be estimated based on the carrier frequency of the pulse. Guan et al. use 1000 time-steps per optical cycle (in small Krylov basis). We can use much less. Demekhin used 50pts/cycle
    # 1050 nm = 1.179 eV = 285 THz -> 1 optical cycle = 3.5 fs
    print("Electric field carrier frequency = "+str("%10.3f"%(params['omega']*1.0e-12))+" THz")
    print("Electric field oscillation period (optical cycle) = "+str("%10.3f"%(1.0e15/params['omega']))+" fs")
    print("suggested time-step for field linear frequency = "+str("%12.3f"%(params['omega']/1e12))+" THz is: " + str("%10.2f"%(opt_cycle/suggested_no_pts_per_cycle )) +" as")

    params['omega'] *= 2.0 * np.pi # linear to angular frequency
    params['omega'] /= 4.13e16 #Hz to a.u.
    frequency_units = "nm" #we later convert all units to atomic unit

    #params['E0'] = 1.0e9 #V/cm
    field_units = "V/cm"

    #convert from W/cm^2 to V/cm
    epsilon0=8.85e-12
    intensity = 7e14 #7e16 #W/cm^2 #peak intensity
    field_strength = np.sqrt(intensity/(vellgt * epsilon0))
    print("field strength")
    print("  %8.2e"%field_strength)
    params['E0'] = field_strength

    # convert time units to atomic units
    time_to_au = {"as" : np.float64(1.0/24.188)}
    # 1a.u. (time) = 2.418 e-17s = 24.18 as

    # convert frequency units to atomic units
    freq_to_au = {"nm" : np.float64(0.057/800.0)}
    # 1a.u. (time) = 2.418 e-17s = 24.18 as

    # convert electric field from different units to atomic units
    field_to_au = {"debye" : np.float64(0.393456),
                    "V/cm" :  np.float64(1.0/(5.14220652e+9))}

    #unit conversion
    #params = const.convert_units(params)
    time_to_au = time_to_au[time_units]

    params['tmax'] *= time_to_au 
    params['dt'] *= time_to_au
    params['time_pecd'] *=time_to_au

    #freq_to_au = freq_to_au[frequency_units]
    #params['omega'] *= freq_to_au 
    field_to_au = field_to_au[field_units]
    params['E0'] *= field_to_au 
    # 1a.u. (time) = 2.418 e-17s = 24.18 as
    #field strength in a.u. (1a.u. = 5.1422e9 V/cm). For instance: 5e8 V/cm = 3.3e14 W/cm^2
   
    params['tau'] = 2000.0 #as: pulse duration

    """==== field dictionaries ===="""
    field_CPL = {"function_name": "fieldCPL", "omega": params['omega'], "E0": params['E0'], "CEP0": 0.0, "spherical": True, "typef": "LCPL"}
    field_LP = {"function_name": "fieldLP", "omega": params['omega'], "E0": params['E0'], "CEP0": 0.0}

    # if gaussian width is given: e^-t^2/sigma^2
    # FWHM = 2.355 * sigma/sqrt(2)
    env_gaussian = {"function_name": "envgaussian", "FWHM": 2.355 * params['tau']/np.sqrt(2.0) * time_to_au , "t0": 500.0 }

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
