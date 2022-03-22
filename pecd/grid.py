#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2021 Emil Zak <emil.zak@cfel.de>
#
import numpy as np
import os

import psi4

import constants


#from sympy.core.compatibility import range

def rotate_mol_xyz(params, grid_euler, irun):
    """ Generate raw cartesian coordinates of atoms from Z-matrix,
        followed by shift to the centre of mass,
        followed by rotation by appropriate rotation matrix associated with elements of the Euler angles grid
    """
    """
        Set up properties of the molecule, including atomic masses, geometry, MF embedding
        *)  mol_geometry: for now only supports a dictionary with internal coordinates (bond lengths, angles). 
            Extend later to the option of the cartesian input. 
        *) mol_embedding (string): define the MF embedding, which matches the embedding used for calculating the ro-vibrational wavefunctions.
            Extend later beyond the TROVE wavefunctions. Add own ro-vibrational wavefunctions and embeddings.
    """
    print("generating rotated cartesian coordinates of atoms...")
    print("irun = " + str(irun))
    print("(alpha,beta,gamma) = " + str(grid_euler[irun]))

    if params['molec_name'] == "d2s":

        mol_xyz = np.zeros( (3,3), dtype = float) #
        mol_xyz_rotated = np.zeros( (3,3), dtype = float) #

        # Sx D1x D2x
        # Sy D1y D2y
        # Sz D1z D2z

        ang_au = constants.angstrom_to_au
        # create raw cartesian coordinates from input molecular geometry and embedding
        rSD1 = params['mol_geometry']["rSD1"] * ang_au
        rSD2 = params['mol_geometry']["rSD2"] * ang_au
        alphaDSD = params['mol_geometry']["alphaDSD"] * np.pi / 180.0

        mS = params['mol_masses']["S"]    
        mD = params['mol_masses']["D"]

        mVec = np.array ( [mS, mD, mD])

        Sz = 1.1 * ang_au  #dummy value (angstrom) of z-coordinate for S-atom to place the molecule in frame of reference. Later shifted to COM.

        mol_xyz[2,0] = Sz

        mol_xyz[1,1] = -1.0 * rSD1 * np.sin(alphaDSD / 2.0)
        mol_xyz[2,1] = -1.0 * rSD1 * np.cos(alphaDSD / 2.0 ) + Sz #checked

        mol_xyz[1,2] = 1.0 * rSD2 * np.sin(alphaDSD / 2.0)
        mol_xyz[2,2] = -1.0 * rSD2 * np.cos(alphaDSD / 2.0 ) + Sz #checked


        print("raw cartesian molecular-geometry matrix")
        print(mol_xyz)

        
        print("Centre-of-mass coordinates: ")
        RCM = np.zeros(3, dtype=float)

        for iatom in range(3):
            RCM[:] += mVec[iatom] * mol_xyz[:,iatom]

        RCM /= np.sum(mVec)

        print(RCM)
    
        for iatom in range(3):
            mol_xyz[:,iatom] -= RCM[:] 

        print("cartesian molecular-geometry matrix shifted to centre-of-mass: ")
        print(mol_xyz)
        print("Rotation matrix:")
        rotmat = R.from_euler('zyz', [grid_euler[irun][0], grid_euler[irun][1], grid_euler[irun][2]], degrees=False)
    
        #ALTErnatively use
        #euler_rot(chi, theta, phi, xyz):
    
        #rmat = rotmat.as_matrix()
        #print(rmat)

        for iatom in range(3):
            mol_xyz_rotated[:,iatom] = rotmat.apply(mol_xyz[:,iatom])
            #mol_xyz_rotated[:,iatom] = euler_rot(cgrid_euler[irun][2], grid_euler[irun][1], grid_euler[irun][0], mol_xyz[:,iatom]):
        print("rotated molecular cartesian matrix:")
        print(mol_xyz_rotated)


    elif params['molec_name'] == "n2":
        
        mol_xyz = np.zeros( (3,2), dtype = float) #
        mol_xyz_rotated = np.zeros( (3,2), dtype = float) #

        #  N1x N2x
        #  N1y N2y
        #  N1z N2z

        ang_au = constants.angstrom_to_au

        mol_xyz[2,0] = ang_au * params['mol_geometry']["rNN"] / 2.0 
        mol_xyz[2,1] =  -1.0 * mol_xyz[2,0] 

        print("Rotation matrix:")
        rotmat = R.from_euler('zyz', [grid_euler[irun][0], grid_euler[irun][1], grid_euler[irun][2]], degrees=False)
    
        for iatom in range(2):
            mol_xyz_rotated[:,iatom] = rotmat.apply(mol_xyz[:,iatom])
        print("rotated molecular cartesian matrix:")
        print(mol_xyz_rotated)


    elif params['molec_name'] == "co":
        
        mol_xyz = np.zeros( (3,2), dtype = float) # initial embedding coordinates (0 degrees rotation)
        mol_xyz_rotated = np.zeros( (3,2), dtype = float) #
        mol_xyz_MF= np.zeros( (3,2), dtype = float) #rotated coordinates: molecular frame embedding

        ang_au = constants.angstrom_to_au

        mC = params['mol_masses']["C"]    
        mO = params['mol_masses']["O"]
        rCO = params['mol_geometry']["rCO"] 

        # C = O   in ----> positive direction of z-axis.
        #coordinates for vanishing electric dipole moment in psi4 calculations is z_C = -0.01 a.u. , z_O = 2.14 a.u
        mol_xyz[2,0] = - 1.0 * ang_au * mO * rCO / (mC + mO) #z_C
        mol_xyz[2,1] =  1.0 * ang_au * mC * rCO / (mC + mO) #z_O

        """rotation associated with MF embedding"""
        rotmatMF = R.from_euler('zyz', [0.0, params['mol_embedding'], 0.0], degrees=True)

        for iatom in range(2):
            mol_xyz_MF[:,iatom] = rotmatMF.apply(mol_xyz[:,iatom])
        print("rotated MF cartesian matrix:")
        print(mol_xyz_MF)


        print("Rotation matrix:")
        rotmat = R.from_euler('zyz', [grid_euler[irun][0], grid_euler[irun][1], grid_euler[irun][2]], degrees=False)
    
        for iatom in range(2):
            mol_xyz_rotated[:,iatom] = rotmat.apply(mol_xyz_MF[:,iatom])
        print("rotated molecular cartesian matrix:")
        print(mol_xyz_rotated)

    elif params['molec_name'] == "h":
        
        mol_xyz = np.zeros( (3,1), dtype = float) #
        mol_xyz_rotated = np.zeros( (3,1), dtype = float) #

        #  Hx
        #  Hy 
        #  Hz 

    elif params['molec_name'] == "c":
        
        mol_xyz = np.zeros( (3,1), dtype = float) #
        mol_xyz_rotated = np.zeros( (3,1), dtype = float) #

        #  Cx
        #  Cy 
        #  Cz 

    elif params['molec_name'] == "h2o":
        mol_xyz = np.zeros( (3,1), dtype = float) #
        mol_xyz_rotated = np.zeros( (3,1), dtype = float) #
    else:
        print("Warning: molecule name not found")
        mol_xyz_rotated = np.zeros( (3,1), dtype = float) 
        #exit()

    #veryfiy rotated geometry by plots
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-2.0,2.0)
    ax.set_ylim(-2.0,2.0)
    ax.set_zlim(-2.0,2.0)
    ax.scatter(mol_xyz_rotated[0,:], mol_xyz_rotated[1,:], mol_xyz_rotated[2,:])
    ax.scatter(mol_xyz[0,:], mol_xyz[1,:], mol_xyz[2,:])
    plt.show()
    exit()

    mlab.figure(1, bgcolor=(0, 0, 0), size=(350, 350))
    mlab.clf()

    # The position of the atoms
    atoms_x = mol_xyz_rotated[0,:]
    atoms_y = mol_xyz_rotated[1,:]
    atoms_z = mol_xyz_rotated[2,:]
    axes = mlab.axes(color=(0, 0, 0), nb_labels=5)
    mlab.orientation_axes()
    O = mlab.points3d(atoms_x[1:-1], atoms_y[1:-1], atoms_z[1:-1],
                    scale_factor=3,
                    resolution=20,
                    color=(1, 0, 0),
                    scale_mode='none')

    H1 = mlab.points3d(atoms_x[:1], atoms_y[:1], atoms_z[:1],
                    scale_factor=2,
                    resolution=20,
                    color=(1, 1, 1),
                    scale_mode='none')

    H2 = mlab.points3d(atoms_x[-1:], atoms_y[-1:], atoms_z[-1:],
                    scale_factor=2,
                    resolution=20,
                    color=(1, 1, 1),
                    scale_mode='none')

    # The bounds between the atoms, we use the scalar information to give
    # color
    mlab.plot3d(atoms_x, atoms_y, atoms_z, [1, 2, 1],
                tube_radius=0.4, colormap='Reds')


    mlab.show()
    """

    return mol_xyz_rotated




def read_leb_quad(scheme, path):
    sphgrid = []
    #print("reading Lebedev grid from file:" + "/lebedev_grids/"+str(scheme)+".txt")

    fl = open( path + "lebedev_grids/" + str(scheme) + ".txt", 'r' )
    #/gpfs/cfel/cmi/scratch/user/zakemil/PECD/pecd/ for maxwell
    for line in fl:
        words   = line.split()
        phi     = float(words[0])
        theta   = float(words[1])
        w       = float(words[2])
        sphgrid.append([theta,phi,w])

    sphgrid = np.asarray(sphgrid)
    sphgrid[:,0:2] = np.pi * sphgrid[:,0:2] / 180.0 #to radians

    """
    xx = np.sin(sphgrid[:,1])*np.cos(sphgrid[:,0])
    yy = np.sin(sphgrid[:,1])*np.sin(sphgrid[:,0])
    zz = np.cos(sphgrid[:,1])
    #Set colours and render
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xx,yy,zz,color="k",s=20)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    plt.tight_layout()
    #plt.show()
    """
    return sphgrid

def GEN_GRID(sph_quad_list, path):
    Gs = []
    for elem in sph_quad_list:
        gs = read_leb_quad(str(elem[3]), path)
        Gs.append( gs )
    return Gs

def sph2cart(r,theta,phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x,y,z

def GEN_XYZ_GRID(Gs,Gr,working_dir):
    grid = []
    r_array = Gr.flatten()
    #print(r_array)
    print("working dir: " + working_dir )
    gridfile = open(working_dir + "grid.dat", 'w')

    for k in range(len(r_array)):
        #print(Gs[k].shape[0])
        for s in range(Gs[k].shape[0]):

            theta   = Gs[k][s,0] 
            phi     = Gs[k][s,1]
            r       = r_array[k]
            x,y,z   = sph2cart(r,theta,phi)

            gridfile.write( " %12.6f"%x +  " %12.6f"%y + "  %12.6f"%z + "\n")
            grid.append([x,y,z])

    return grid

def CALC_ESP_PSI4(dir,params):
    os.chdir(dir)
    psi4.core.be_quiet()
    properties_origin=["COM"]
    #psi4.core.set_output_file(dir, False)
    ang_au = constants.angstrom_to_au
    xS = 0.0 * ang_au
    yS = 0.0 * ang_au
    zS = 0.0 * ang_au

    xH1 = 0.0 * ang_au
    yH1 = 0.1 * ang_au
    zH1 = 1.2 * ang_au

    xH2 = 0.1 * ang_au
    yH2 = 1.4 * ang_au
    zH2 = -0.1 * ang_au

    mol = psi4.geometry("""
    1 2
    noreorient
    units au
    
    S	{0} {1} {2}
    H1	{3} {4} {5}
    H2	{6} {7} {8}

    """.format(xS, yS, zS, xH1, yH1, zH1, xH2, yH2, zH2)
    )

    psi4.set_options({'basis': params['scf_basis'], 'e_convergence': params['scf_enr_conv'], 'reference': params['scf_method']})
    E, wfn = psi4.prop('scf', properties = ["GRID_ESP"], return_wfn = True)
    Vvals = wfn.oeprop.Vvals()
    os.chdir("../")
    return Vvals


def CALC_ESP_PSI4_ROT(dir,params,mol_xyz):
    os.chdir(dir)
    psi4.core.be_quiet()
    properties_origin=["COM"] #[“NUCLEAR_CHARGE”] or ["COM"] #here might be the shift!
    ang_au = constants.angstrom_to_au

    if params['molec_name'] == "d2s":
        mol = psi4.geometry("""
        1 2
        noreorient
        units au
        nocom
        S	{0} {1} {2}
        H	{3} {4} {5}
        H	{6} {7} {8}

        """.format( mol_xyz[0,0], mol_xyz[1,0], mol_xyz[2,0],
                    mol_xyz[0,1], mol_xyz[1,1], mol_xyz[2,1],
                    mol_xyz[0,2], mol_xyz[1,2], mol_xyz[2,2],)
        )
    
    if params['molec_name'] == "cmethane":
        mol = psi4.geometry("""
        1 2
        units au
        C
        H  1 CH1
        H  1 CH2  2 HCH
        H  1 CH3  2 HCH    3  120.0
        H  1 CH4  2 HCH    3  240.0

        CH1    = {0}
        CH2    = {1}
        CH3    = {2}
        CH4    = {3}
        HCH    = 109.471209
        """.format( params['mol_geometry']['r1'], params['mol_geometry']['r2'],
                    params['mol_geometry']['r3'], params['mol_geometry']['r4'])
        )
    


    elif params['molec_name'] == "n2":
        mol = psi4.geometry("""
        1 2
        noreorient
        units au
        nocom
        N	{0} {1} {2}
        N	{3} {4} {5}

        """.format( mol_xyz[0,0], mol_xyz[1,0], mol_xyz[2,0],
                    mol_xyz[0,1], mol_xyz[1,1], mol_xyz[2,1])
        )


    elif params['molec_name'] == "co":
        mol = psi4.geometry("""
        1 2
        noreorient
        units au
        nocom
        C	{0} {1} {2}
        O	{3} {4} {5}

        """.format( mol_xyz[0,0], mol_xyz[1,0], mol_xyz[2,0],
                    mol_xyz[0,1], mol_xyz[1,1], mol_xyz[2,1])
        )

    elif params['molec_name'] == "ocs":
        mol = psi4.geometry("""
        1 2
        noreorient
        units au
        
        O	{0} {1} {2}
        C	{3} {4} {5}
        S	{6} {7} {8}

        """.format( mol_xyz[0,0], mol_xyz[1,0], mol_xyz[2,0],
                    mol_xyz[0,1], mol_xyz[1,1], mol_xyz[2,1],
                    mol_xyz[0,2], mol_xyz[1,2], mol_xyz[2,2],)

        )


    elif params['molec_name'] == "h2o":
        mol = psi4.geometry("""
        1 2
        noreorient
        units au
        
        O	{0} {1} {2}
        H	{3} {4} {5}
        H	{6} {7} {8}

        """.format( 0.0, 0.0, 0.0,
                    0.0, 1.0, 1.0,
                    0.0, -2.0, 2.0,)
        )

    elif params['molec_name'] == "h":
        mol = psi4.geometry("""
        1 1
        noreorient
        units au
        nocom
        H	{0} {1} {2}

        """.format( 0.0, 0.0, 0.0)
        )

    elif params['molec_name'] == "c":
        mol = psi4.geometry("""
        1 2
        noreorient
        units au
        
        C	{0} {1} {2}

        """.format( 0.0, 0.0, 0.0)
        )

    psi4.set_options({'basis': params['scf_basis'], 'e_convergence': params['scf_enr_conv'], 'reference': params['scf_method']})
    E, wfn = psi4.prop('scf', properties = ["GRID_ESP"], return_wfn = True)
    Vvals = wfn.oeprop.Vvals()
    os.chdir("../")
    return Vvals



    #D2S geometry (NIST)
    # r(S-D) = 1.336 ang
    #angle = 92.06
    """
    units angstrom
    S	    0.0000	0.0000	0.1031
    H@2.014	0.0000	0.9617	-0.8246
    H@2.014	0.0000	-0.9617	-0.8246"""

