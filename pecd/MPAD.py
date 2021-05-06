
"""=== Fourier transform and momentum distributions ==="""
def momentum_pad_expansion(self,psi):
    """
    Main function for the calculation of photo-electron 3D distributions in real space and momentum space. 
    Spherical expansion of momentum PAD. 

    Returns: array wLM(k): spherical expansion coeffs for momentum space probability on a grid of k-vector lengths.
    """
    WLM_array = [] #array of converged spherical expansion coefficients: L,M,WLM(k0), WLM(k1), WLM(k2), ..., WLM(kmax)

    """ read in spherical quadrature schemes """
    spherical_schemes = []
    for elem in list(quadpy.u3.schemes.keys()):
        if 'lebedev' in elem:
            spherical_schemes.append(elem)

    """ set up indices for WLM """
    Lmax = params['pecd_lmax']
    Lmin = 0
    WLM_quad_tol = params['WLM_quad_tol'] 
    #create list of basis set indices
    anglist = []
    for l in range(Lmin,Lmax+1):
        for m in range(-l,l+1):
            anglist.append([l,m])

    WLM =  np.zeros(shape=(len(anglist)),dtype=complex)
    WLM_prev = np.zeros(shape=(len(anglist)),dtype=complex)

    """ loop over the k-vector grid """
    for k in params['k-grid']:
        print("k = " +str(k))
        start_time = time.time()   

        print("checking convergence of WLM(k) vs. Lebedev quadrature level")
        for scheme in spherical_schemes[3:6]: #skip 003a,003b,003c rules
            print("generating Lebedev grid at level: " +str(scheme))
            G = self.gen_gft(scheme)

            """ 2) Spherical expansion of |FT[psi]|^2 into WLM(k) """     
            il = 0
            for L,M in anglist:
                y = 0.0 + 1j * 0.0
                """ calculate WLM """
                for xi in range(len(G[:,0])): #make this vectorized. Means make
                    y += G[xi,2] * sph_harm(M, L, G[xi,1]+np.pi , G[xi,0] ) * np.abs(self.FTpsi(psi,k,G,xi))**2
                WLM[il] = 4.0 * np.pi * y
                il +=1
    
            print("WLM:")
            print(WLM)
            print("WLM_prev:")
            print(WLM_prev)
            print("differences:")
            print(WLM - WLM_prev)

            #check for convergence
            diff = np.abs(WLM - WLM_prev)

            if (np.any(diff>WLM_quad_tol)):
                print(str(scheme)+" WLM convergence not reached") 
                WLM_prev = np.copy(WLM)

            elif (np.all(diff<WLM_quad_tol)):     
                print(str(scheme)+" " +"k = " +str(k) + ", WLM convergence reached!!!")
                break

            #if no convergence reached raise warning
            if (scheme == spherical_schemes[len(spherical_schemes)-1] and np.any(diff>WLM_quad_tol)):
                print("WARNING: convergence at tolerance level = " + str(WLM_quad_tol) + " not reached for all considered quadrature schemes")
        end_time = time.time()
        print("Execution time for calculation of WLMs at single k: " + str(end_time-start_time))
        exit()
    #print("array of spherical expansion coefficients")
    #print(WLM_array)

    return WLM_array

def wLM(self,k,L,M,myscheme,ft_interp_real,ft_interp_imag):
    #calculate wLM(k) expansion coefficients in spherical harmonics for given value of k
    start_time = time.time()    
    """
    Symbols: 
            theta_phi[0] = theta in [0,pi]
            theta_phi[1] = phi  in [-pi,pi]
            sph_harm(m1, l1,  theta_phi[1]+np.pi, theta_phi[0])) means that we put the phi angle in range [0,2pi] and the  theta angle in range [0,pi] as required by the scipy special funciton sph_harm
    """
    val = myscheme.integrate_spherical(lambda theta_phi: np.conjugate(sph_harm(M, L,  theta_phi[1]+np.pi, theta_phi[0])) * np.abs(ft_interp_real([k,theta_phi[0],theta_phi[1]+np.pi])+1j*ft_interp_imag([k,theta_phi[0],theta_phi[1]+np.pi]))**2 )

    end_time = time.time()
    print("Execution time for calculation of single WLM: " + str(end_time-start_time))

    return val * np.sqrt((2 * L +1)/(4 * np.pi) )
    
def pecd(self,wlm_array):
    return  2*wlm_array[2,2].real     

def gen_all_gfts(self):
    """ read in spherical quadrature schemes """
    spherical_schemes = []
    for elem in list(quadpy.u3.schemes.keys()):
        if 'lebedev' in elem:
            spherical_schemes.append(elem)

    grids = []

    for scheme in spherical_schemes[3:]: #skip 003a,003b,003c rules
        print("generating Lebedev grid at level: " +str(scheme))
        grids.append(self.gen_gft(scheme))
    
    return grids



def gen_gft(self,scheme):
    "generate spherical grid for evaluation of FT"
    if params['FT_output'] == "lebedev_grid":
        sphgrid = []
        print("reading Lebedev grid from file:" + "/lebedev_grids/"+str(scheme)+".txt")
        fl = open("./lebedev_grids/"+str(scheme)+".txt",'r')
        for line in fl:
            words = line.split()
            phi = float(words[0])
            theta = float(words[1])
            w = float(words[2])
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

def FTpsi(self,coeffs,k,G,xi):
    #generate gauss-Lobatto global grid
    rbas = radbas(params['nlobatto'], params['nbins'], params['binwidth'], params['rshift'])
    nlobatto = params['nlobatto']
    rgrid  = rbas.r_grid()
    #generate Gauss-Lobatto quadrature
    xlobatto=np.zeros(nlobatto)
    wlobatto=np.zeros(nlobatto)
    xlobatto,wlobatto=rbas.gauss_lobatto(nlobatto,14)
    wlobatto=np.array(wlobatto)
    xlobatto=np.array(xlobatto) # convert back to np arrays

    mymap = mapping(int(params['lmin']), int(params['lmax']), int(params['nbins']), int(params['nlobatto']))
    maparray, Nbas = mymap.gen_map()

    # 1) Evaluate psi(r,theta,phi) on a cartesian grid
    x = np.linspace( params['rshift'], params['binwidth'] * params['nbins'] ,   params['nkpts'],dtype=float)
    y = np.linspace( params['rshift'], params['binwidth'] * params['nbins'] ,   params['nkpts'],dtype=float)
    z = np.linspace( params['rshift'], params['binwidth'] * params['nbins'] ,   params['nkpts'],dtype=float)
    X,Y,Z = np.meshgrid(x,y,z,indexing='ij')

    psi_grid = np.zeros([np.size(x) * np.size(y) * np.size(z)], dtype = complex)              
    #print("Shape of psi_grid:" + str(psi_grid.shape))
    #print(np.shape(self.cart2sph(X,Y,Z)))


    """ set up radial quadratures """
    if params['schemeFT_rad'][0] == "Gauss-Laguerre":
        Nquad = params['schemeFT_rad'][1] 
        x,w = roots_genlaguerre(Nquad ,1)
        alpha = 1 #parameter of the G-Lag quadrature
        inv_weight_func = lambda r: r**(-alpha)*np.exp(r)
    elif params['schemeFT_rad'][0] == "Gauss-Hermite":
        Nquad = params['schemeFT_rad'][1] 
        x,w = roots_hermite(Nquad)
        inv_weight_func = lambda r: np.exp(r**2)




    """     #the meshes must be flattened
            r1d = self.cart2sph(X,Y,Z)[0].flatten()
            theta1d = self.cart2sph(X,Y,Z)[1].flatten()
            phi1d = self.cart2sph(X,Y,Z)[2].flatten()
            print(r1d)"""
    if  params['FT_method'] == "fftn":
        for ielem,elem in enumerate(maparray):
            psi_grid += coeffs[ielem] *  rbas.chi(elem[2],elem[3],r1d,rgrid,wlobatto) *   sph_harm(elem[1], elem[0],  phi1d, theta1d) 
        #coeffs[ielem] #*
        psift = np.fft.fftn(psi_grid)

    elif  params['FT_method'] == "quadrature":   

        print("calculating FT using quadratures")

        if params['FT_output'] == "lebedev_grid":

            print("calculating  FT on Lebedev grid")

            kmesh = k
            theta_k = G[xi,0]
            phi_k = G[xi,1]

            print(str((kmesh,theta_k,phi_k)))

            Nrad = (params['nlobatto']-1) * params['nbins'] -1

            ftpsi = 0.0 + 1j * 0.0
            for s in range(len(x)):
            
                """ generate converged I_lm(r_s,k,theta_k,phi_k) matrix """
                Imat = self.Ilm(x[s],kmesh,theta_k,phi_k)

                #for ielem,elem in enumerate(maparray):
                vs = 0.0 + 1j * 0.0
                for beta in range(len(Imat)):
                    g = 0.0 + 1j * 0.0
                    for alpha in range(beta*Nrad,(beta+1)*Nrad ):
                        g+= coeffs[alpha] * rbas.chi(maparray[alpha][2],maparray[alpha][3],x[s],rgrid,wlobatto) 

                    vs += Imat[beta] *  g


                ftpsi += w[s] * x[s] * vs



            print(ftpsi)

    return ftpsi

def Ilm(self,r,kmesh,theta_k,phi_k):
    if params['test_conv_FT_ang'] == True:
        print("checking for convergence the angular spectral part of FT vs. quadrature level")
        spherical_schemes = []
        for elem in list(quadpy.u3.schemes.keys()):
            if 'lebedev' in elem:
                spherical_schemes.append(elem)
        #print("Available schemes: " + str(spherical_schemes))

        lmax = params['lmax']
        lmin = params['lmin']
        quad_tol = params['quad_tol']

        """calculate the  I(r;k,theta_k,phi_k)  = Y_lm(theta,phi) e**(i * k * r) d Omega integral """

        #create list of basis set indices
        anglist = []
        for l in range(lmin,lmax+1):
            for m in range(-l,l+1):
                anglist.append([l,m])
        val =  np.zeros(shape=(len(anglist)),dtype=complex)
        val_prev = np.zeros(shape=(len(anglist)),dtype=complex)
        for scheme in spherical_schemes[3:]: #skip 003a,003b,003c rules
            G = self.gen_gft(scheme)

            il = 0
            for l,m in anglist:
                y = 0.0 + 1j * 0.0
                for xi in range(len(G[:,0])):

                    y += G[xi,2] * sph_harm(m, l,G[xi,1]+np.pi , G[xi,0] ) \
                    * np.exp( -1j * ( kmesh * np.sin(theta_k) *\
                        np.cos(phi_k) * r * np.sin(G[xi,0]) * np.cos(G[xi,1]+np.pi ) + kmesh * \
                            np.sin(theta_k) * np.sin(phi_k) * r * np.sin(G[xi,0]) * np.sin(G[xi,1]+np.pi ) \
                                + kmesh * np.cos(theta_k) * r * np.cos(G[xi,0]))) 
                val[il] = 4.0 * np.pi * y
                il +=1
            #print("differences:")
            #print(val - val_prev)

            #check for convergence
            diff = np.abs(val - val_prev)

            if (np.any(diff>quad_tol)):
                #print(str(scheme)+" convergence not reached") 
                val_prev = np.copy(val)

            elif (np.all(diff<quad_tol)):     
                print(str(scheme)+" convergence reached!!!")
                return val


            #if no convergence reached raise warning
            if (scheme == spherical_schemes[len(spherical_schemes)-1] and np.any(diff>quad_tol)):
                print("WARNING: convergence at tolerance level = " + str(quad_tol) + " not reached for all considered quadrature schemes")


    def FTpsicart_interpolate(self,psift,x,y,z):
        print(np.shape(psift))
        print(np.shape(x))
        return interpolate.RegularGridInterpolator((x,y,z), psift.real), interpolate.RegularGridInterpolator((x,y,z), psift.real)

    def interpolate_FT(self, psi):
        """this function returns the interpolant of the fourier transform of the total wavefunction, for later use in spherical harmonics decomposition and plotting"""
        """ For now we work on 2D grid of (k,theta), at fixed value of phi """
        phi_k_0 = np.pi/12
        k_1d = np.linspace( params['momentum_range'][0], params['momentum_range'][1], params['n`kpts'])
        thetak_1d = np.linspace(0,   np.pi,  2*20) # 
        #phik_1d   = np.linspace(0, 2*np.pi, 2*181) # 

        #generate gauss-Lobatto global grid
        rbas = radbas(params['nlobatto'], params['nbins'], params['binwidth'], params['rshift'])
        nlobatto = params['nlobatto']
        nbins = params['nbins']
        rgrid  = rbas.r_grid()
        #generate Gauss-Lobatto quadrature
        xlobatto=np.zeros(nlobatto)
        wlobatto=np.zeros(nlobatto)
        xlobatto,wlobatto=rbas.gauss_lobatto(nlobatto,14)
        wlobatto=np.array(wlobatto)
        xlobatto=np.array(xlobatto) # convert back to np arrays

        mymap = mapping(int(params['lmin']), int(params['lmax']), int(params['nbins']), int(params['nlobatto']))
        maparray, Nbas = mymap.gen_map()

        if params['FT_method'] == "quadrature": #set up quadratures
            #print("Using quadratures to calculate FT of the wavefunction")

            FTscheme_ang = quadpy.u3.schemes[params['schemeFT_ang']]()

            if params['schemeFT_rad'][0] == "Gauss-Laguerre":
                Nquad = params['schemeFT_rad'][1] 
                x,w = roots_genlaguerre(Nquad ,1)
                alpha = 1 #parameter of the G-Lag quadrature
                inv_weight_func = lambda r: r**(-alpha)*np.exp(r)
            elif params['schemeFT_rad'][0] == "Gauss-Hermite":
                Nquad = params['schemeFT_rad'][1] 
                x,w = roots_hermite(Nquad)
                inv_weight_func = lambda r: np.exp(r**2)


        k_mesh, theta_mesh= np.meshgrid(k_1d, thetak_1d )
        values = self.FTpsi(k_mesh,theta_mesh,phi_k_0,wlobatto,rgrid,maparray,psi,FTscheme_ang,x,w,inv_weight_func,rbas)

        #print(values)
        #print(np.shape(values))
        #print(np.shape(theta_mesh))
        #print(np.shape(k_mesh))
        #exit()
        FTinterp = interpolate.interp2d(k_mesh, theta_mesh, values.real, kind='cubic')
        #FTinterp = interpolate.RectBivariateSpline(k_1d,thetak_1d,values.real)
        return FTinterp
