import numpy as np
import h5py
import re
import itertools
import wigner
from wigner.wiglib import DJmk, DJ_m_k
import sys



def read_coefficients(coef_file, coef_thresh=1.0e-16):
    """Reads Richmol coefficients file
    """
    print("\nRead Richmol states' coefficients file", coef_file)
    states = []
    fl = open(coef_file, "r")
    for line in fl:
        w = line.split()
        jrot = int(w[0])
        id = int(w[1])
        ideg = int(w[2])
        enr = float(w[3])
        nelem = int(w[4])
        coef = []
        vib = []
        krot = []
        for ielem in range(nelem):
            c = float(w[5+ielem*4])
            im = int(w[6+ielem*4])
            if abs(c)**2<=coef_thresh: continue
            coef.append(c*{0:1,1:1j}[im])
            vib.append(int(w[7+ielem*4]))
            krot.append(int(w[8+ielem*4]))
        states.append({"j":jrot,"id":id,"ideg":ideg,"coef":coef,"v":vib,"k":krot,"enr":enr})
    fl.close()
    return states


def read_wavepacket(coef_file, coef_thresh=1.0e-16):
    """Reads Richmol wavepacket coefficients file
    """
    print("\nRead Richmol wavepacket file", coef_file)
    time = []
    quanta = []
    coefs = []
    h5 = h5py.File(coef_file, mode='r')
    for key in h5.keys():
        st = re.sub(r"results_t_", "", key)
        t = float(re.sub("_", ".", st))
        q = h5[key]["quanta_t_"+st].value
        c = h5[key]["coef_t_"+st].value
        quanta.append(q[abs(c)**2>coef_thresh,:])
        coefs.append(c[abs(c)**2>coef_thresh])
        time.append(t)
    h5.close()
    return time, coefs, quanta


def calc_rotdens(grid_3d, coef_file, wavepacket_file):
    #calc rotdens at a point
  
    states = read_coefficients(coef_file, coef_thresh=1.0e-16)
    time, coefs, quanta = read_wavepacket(wavepacket_file, coef_thresh=1.0e-16)

  
    npoints_3d = grid_3d.shape[0]
    # mapping between wavepacket and rovibrational states

    ind_state = []
    for q in quanta:
        j = q[1] #q[0] = M?
        id = q[2]
        ideg = q[3]
        istate = [(state["j"],state["id"],state["ideg"]) for state in states].index((j,id,ideg)) #find in which position in the states list we have quanta j, id, ideg - from the current wavepacket
	#state["j"] - this is how we refer to elements of a dictionary
        ind_state.append(istate) #at each time we append an array of indices which locate the current wavepacket in the states dictionary


    # lists of J and m quantum numbers

    jlist = list(set(j for j in quanta[:,1]))
    mlist = []
    for j in jlist:
        mlist.append(list(set(m  for m,jj in zip(quanta[:,0],quanta[:,1]) if jj==j)))
    print("List of J-quanta:", jlist)
    print("List of m-quanta:", mlist)


    # precompute symmetric-top functions on a 3D grid of Euler angles for given J, m=J, and k=-J..J

    print("\nPrecompute symmetric-top functions...")
    symtop = []
    for J,ml,ij in zip(jlist,mlist,range(len(jlist))):
        print("J = ", J)
        Jfac = np.sqrt((2*J+1)/(8*np.pi**2))
        symtop.append([])
        for m in ml:
            print("m = ", m)
            wig = wigner.wiglib.DJ_m_k(int(J), int(m), grid_3d[:,:]) #grid_3d= (3,npoints_3d). Returns wig = array (npoints, 2*J+1) for each k,  #wig Contains values of D-functions on grid, 
            #D_{m,k}^{(J)} = wig[ipoint,k+J], so that the range for the second argument is 0,...,2J
       
            symtop[ij].append( np.conj(wig) * Jfac )
    print("...done")


    # compute rotational density

    vmax = max([max([v for v in state["v"]]) for state in states])
    func = np.zeros((npoints_3d,vmax+1), dtype=np.complex128)
    tot_func = np.zeros((npoints_3d,vmax+1), dtype=np.complex128)
    dens = np.zeros(npoints_3d, dtype=np.complex128)

    for q,cc,istate in zip(quanta,coefs,ind_state): #loop over coefficients in the wavepacket

        m = q[0]
        j = q[1]
        state = states[istate]

        ind_j = jlist.index(j)
        ind_m = mlist[ind_j].index(m)

        # primitive rovibrational function on Euler grid
        func[:,:] = 0
        for v,k,c in zip(state["v"],state["k"],state["coef"]): #loop over coefficients of primitive symmetric top functions comprising individual components of the wavepacket
            func[:,v] += c * symtop[ind_j][ind_m][:,k+int(j)] #identical rotational functions are used for all vibrational states.
            # The only contribution from vibrations is encoded in v-dependent coefficients. 

        # total function
        tot_func[:,:] += func[:,:] * cc

    # reduced rotational density on Euler grid
    dens = np.einsum('ij,ji->i', tot_func, np.conj(tot_func.T)) * np.sin(grid_3d[1,:]) 
    #tensor contraction: element-wise multuplication of tot_func and transpose of np.conj(tot_func.T)) * np.sin(grid_3d[1,ipoint0:ipoint1] and we take diagonal elements of the output.
    # This is to remove the vibrational index. 

    return grid_3d, dens


def rotdens(npoints, nbatches, ibatch, states, quanta, coefs):
    """
    """

    # generate 3D grid of Euler angles

    npt = int(npoints**(1/3)) # number of points in 1D
    alpha = list(np.linspace(0, 2*np.pi, num=npt, endpoint=True))
    beta  = list(np.linspace(0, np.pi, num=npt, endpoint=True))
    gamma = list(np.linspace(0, 2*np.pi, num=npt, endpoint=True))
    grid = [alpha, beta, gamma]
    grid_3d = np.array(list(itertools.product(*grid))).T #cartesian product of [alpha,beta,gamma]
    npoints_3d = grid_3d.shape[1]
    print("\nTotal number of points in 3D grid of Euler angles:", npoints_3d, " and the shape of the 3D grid array is:    ", grid_3d.shape)

    # process only smaller batch of points
    npt = int(npoints_3d / nbatches)
    ipoint0 = npt*ibatch
    if ibatch==nbatches-1:
        ipoint1 = npoints_3d
    else:
        ipoint1 = ipoint0 + npt - 1
    npoints_3d = ipoint1 - ipoint0
    print(ibatch, "-batch number of points in 3D grid of Euler angles:", npoints_3d, "[", ipoint0, "-", ipoint1, "]")


    # mapping between wavepacket and rovibrational states

    ind_state = []
    for q in quanta:
        j = q[1] #q[0] = M?
        id = q[2]
        ideg = q[3]
        istate = [(state["j"],state["id"],state["ideg"]) for state in states].index((j,id,ideg)) #find in which position in the states list we have quanta j, id, ideg - from the current wavepacket
	#state["j"] - this is how we refer to elements of a dictionary
        ind_state.append(istate) #at each time we append an array of indices which locate the current wavepacket in the states dictionary


    # lists of J and m quantum numbers

    jlist = list(set(j for j in quanta[:,1]))
    mlist = []
    for j in jlist:
        mlist.append(list(set(m  for m,jj in zip(quanta[:,0],quanta[:,1]) if jj==j)))
    print("List of J-quanta:", jlist)
    print("List of m-quanta:", mlist)


    # precompute symmetric-top functions on a 3D grid of Euler angles for given J, m=J, and k=-J..J

    print("\nPrecompute symmetric-top functions...")
    symtop = []
    for J,ml,ij in zip(jlist,mlist,range(len(jlist))):
        print("J = ", J)
        Jfac = np.sqrt((2*J+1)/(8*np.pi**2))
        symtop.append([])
        for m in ml:
            print("m = ", m)
            wig = DJ_m_k(int(J), int(m), grid_3d[:,ipoint0:ipoint1]) #grid_3d= (3,npoints_3d). Returns wig = array (npoints, 2*J+1) for each k,  #wig Contains values of D-functions on grid, 
            #D_{m,k}^{(J)} = wig[ipoint,k+J], so that the range for the second argument is 0,...,2J
       
            symtop[ij].append( np.conj(wig) * Jfac )
    print("...done")


    # compute rotational density

    vmax = max([max([v for v in state["v"]]) for state in states])
    func = np.zeros((npoints_3d,vmax+1), dtype=np.complex128)
    tot_func = np.zeros((npoints_3d,vmax+1), dtype=np.complex128)
    dens = np.zeros(npoints_3d, dtype=np.complex128)

    for q,cc,istate in zip(quanta,coefs,ind_state): #loop over coefficients in the wavepacket

        m = q[0]
        j = q[1]
        state = states[istate]

        ind_j = jlist.index(j)
        ind_m = mlist[ind_j].index(m)

        # primitive rovibrational function on Euler grid
        func[:,:] = 0
        for v,k,c in zip(state["v"],state["k"],state["coef"]): #loop over coefficients of primitive symmetric top functions comprising individual components of the wavepacket
            func[:,v] += c * symtop[ind_j][ind_m][:,k+int(j)] #identical rotational functions are used for all vibrational states.
            # The only contribution from vibrations is encoded in v-dependent coefficients. 

        # total function
        tot_func[:,:] += func[:,:] * cc

    # reduced rotational density on Euler grid
    dens = np.einsum('ij,ji->i', tot_func, np.conj(tot_func.T)) * np.sin(grid_3d[1,ipoint0:ipoint1]) 
    #tensor contraction: element-wise multuplication of tot_func and transpose of np.conj(tot_func.T)) * np.sin(grid_3d[1,ipoint0:ipoint1] and we take diagonal elements of the output.
    # This is to remove the vibrational index. 

    return grid_3d, dens, [ipoint0,ipoint1]



if __name__ == "__main__":

    coef_file = sys.argv[1]
    wavepacket_file = sys.argv[2]
    time0 = round(float(sys.argv[3]),1)
    dens_name = sys.argv[4]

    states = read_coefficients(coef_file, coef_thresh=1.0e-06)
    time, coefs, quanta = read_wavepacket(wavepacket_file, coef_thresh=1.0e-06)
    #time = np.asarray(time)
    #coefs = np.asarray(coefs)
    quanta = np.asarray(quanta)
    for itime in range(0,10):
    	print(time[itime],quanta[itime],coefs[itime])

    # Cartesian coordinates of D2S

    xyz = np.array([[  0.00000000,        0.00000000,        0.10358697],
                    [ -0.96311715,        0.00000000,       -0.82217544],
                    [  0.96311715,        0.00000000,       -0.82217544]], dtype=np.float64)

    # polarizability axes: small, middle, main (c = y, b = z, a = x)

    #xyz = np.array([[0,1,0],
    #                [0,0,1],
    #                [1,0,0]], dtype=np.float64)

    time_index = [round(t,1) for t in time].index(time0)

    #for itime in range(len(time)):
    #    print(itime,time[itime])
    #sys.exit()

    # manual state entry
    #quanta0 = np.array([[-10,10,1,1]])
    #coefs0 = [1.0]

    for itime in [time_index]:

        print(itime, time[itime])

        # compute rotational density

        npoints = 1000000
        nbatches = 1

        for ibatch in range(nbatches):
            grid_3d, dens, [ipt1,ipt2] = rotdens(npoints, nbatches, ibatch, states, quanta[itime], coefs[itime])
            #grid_3d, dens, [ipt1,ipt2] = rotdens(npoints, nbatches, ibatch, states, quanta0, coefs0)
            if ibatch==0:
                dens_3d = np.zeros(grid_3d.shape[1], dtype=np.float64)
            dens_3d[ipt1:ipt2] = dens[:]

        # do monte-carlo

        monte_carlo(grid_3d, dens_3d, xyz, dens_name, npoints=100000)
