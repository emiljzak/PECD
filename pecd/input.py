import numpy as np
def gen_input():

    params = {}
    time_to_au =  np.float64(1.0/24.188)

    freq_to_au = np.float64(0.057/800.0)

    field_to_au =  np.float64(1.0/(5.14220652e+9))
    """====basis set parameters===="""
    params['nlobatto'] = 8
    params['nbins'] = 1 
    params['binwidth'] = 3.0
    params['rshift'] = 1e-5 #rshift must be chosen such that it is non-zero and does not cover significant probability density region of any eigenfunction.
    params['lmin'] = 0
    params['lmax'] = 2
    
    """====runtime controls===="""
    params['method'] = "dynamic_direct" #static: solve time-independent SE for a given potential; dynamic_direct, dynamic_lanczos
    params['basis'] = "prim" # or adiab
    params['potential'] = "pot_hydrogen" # 1) pot_diagonal (for tests); 2) pot_hydrogen; 3) pot_null
    params['scheme'] = "lebedev_019" #angular integration rule
    params['int_rep_type'] = 'spherical' #representation of the interaction potential (spherical or cartesian ): for now only used in calculations of instantaneous electron wavepacket energy.
    params['t0'] = 0.0 
    params['tmax'] = 10.0 
    params['dt'] = 3.0 
    time_units = "as"


    """===== post-processing and analysis ====="""
    params['wavepacket_file'] = "wavepacket.dat" #filename into which the time-dependent wavepacket is saved
    params['plot_modes'] = {"single_shot": False, "animation": False}

    params['plot_types'] = { "radial": True,
                             "angular": False,
                             "r-radial_angular": False, 
                             "k-radial_angular": False} #decide which of the available observables you wish to plot

    params['plot_controls'] = { "plotrate": 1, 
                                "plottimes": [0.0,20.0,40.0,60.0,80.0,100.0,200.0,300.0,600.0,700.0,800.0,900.0,1000.0],
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
    params['FT_method'] = "quadrature" #method of calculating the FT of the wavefunction: quadrature or fftn
    params['schemeFT_ang'] = "lebedev_025" #angular integration rule for calculating FT using the quadratures method
    params['schemeFT_rad'] = ("Gauss-Hermite",20) #quadrature type for projection of psi(t) onto the lobatto basis: Gauss-Laguerre, Gauss-Hermite
    params['pecd_lmax'] = 2 #maximum angular momentum of the expansion into spherical harmonics of the momentum probability function
    params['calc_inst_energy'] = False #calculate instantaneous energy of a free-electron wavepacket?
    params['momentum_range'] = [0.0, 10.0] #range of momenta for the electron (effective radial range for the Fourier transform of the total wavefunction). Note that E = 1/2 * k^2, so it is easily convertible to photo-electron energy range
    params['calculate_pecd'] = True #calculate FT of the wavefunction and expand it into spherical harmonics and calculate PECD?
    params['time_pecd'] = 3.0 #at what time (in a.u.) do we want to calculate PECD?


    """====initial state====""" 
    params['ini_state'] = "eigenvec" #spectral_manual, spectral_file, grid_1d_rad, grid_2d_sph,grid_3d,solve (solve static problem in Lobatto basis), eigenvec (eigenfunciton of static hamiltonian)
    params['ini_state_quad'] = ("Gauss-Laguerre",60) #quadrature type for projection of the initial wavefunction onto lobatto basis: Gauss-Laguerre, Gauss-Hermite
    params['ini_state_file_coeffs'] = "wf0coeffs.txt" # if requested: name of file with coefficients of the initial wavefunction in our basis
    params['ini_state_file_grid'] = "wf0grid.txt" #if requested: initial wavefunction on a 3D grid of (r,theta,phi)
    params['nbins_iniwf'] = 1 #number of bins in a reduced-size grid for generating the initial wavefunction by diagonalizing the static hamiltonian
    params['eigenvec_id'] = 2 #id (in ascending energy order) of the eigenvector of the static Hamiltonian to be used as the initial wavefunction for time-propagation. Beginning 0.
    params['save_ini_wf'] = False #save initial wavefunction generated with eigenvec option to a file (spectral representation)


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
    print("Electric field carrier frequency = "+str(params['omega']*1.0e-12)+" THz")
    print("Electric field oscillation period (optical cycle) = "+str(1.0e15/params['omega'])+" fs")
    print("suggested time-step for field linear frequency = "+str("%12.3f"%(params['omega']/1e12))+" THz is: " + str(opt_cycle/suggested_no_pts_per_cycle ) +" as")


    params['omega'] *= 2.0 * np.pi # linear to angular frequency
    params['omega'] /= 4.13e16 #Hz to a.u.
    frequency_units = "nm" #we later convert all units to atomic unit


    #params['E0'] = 1.0e9 #V/cm
    field_units = "V/cm"

    #convert from W/cm^2 to V/cm
    epsilon0=8.85e-12
    intensity = 7e16 #7e16 #W/cm^2 #peak intensity
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
   

    """==== field dictionaries ===="""
    field_CPL = {"function_name": "fieldCPL", "omega": params['omega'], "E0": params['E0'], "CEP0": 0.0, "spherical": False, "typef": "LCPL"}
    field_LP = {"function_name": "fieldLP", "omega": params['omega'], "E0": params['E0'], "CEP0": 0.0}

    # if gaussian width is given: e^-t^2/sigma^2
    # FWHM = 2.355 * sigma/sqrt(2)

    env_gaussian = {"function_name": "envgaussian", "FWHM": 2.355 * 2000.0/np.sqrt(2.0) * time_to_au , "t0": 5000.0 * time_to_au }

    params['field_form'] = "analytic" #or numerical
    params['field_type'] = field_LP 
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
