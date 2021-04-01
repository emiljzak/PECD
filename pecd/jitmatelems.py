from numba import jit,prange
import numpy as np
import input
from matelem import mapping
from basis import radbas
import time
from scipy.special import sph_harm
from scipy import interpolate
import quadpy
from pyspherical import spin_spherical_harmonic

def prep(params):
    print("\n")
    print("1) Generating radial grid")
    print("\n")
    #we need to create rgrid only once, i.e. we use a static grid
    rbas = radbas(params['nlobatto'], params['nbins'], params['binwidth'], params['rshift'])
    rgrid = rbas.r_grid()
    nlobatto = params['nlobatto']
    #rbas.plot_chi(0.0,params['nbins'] * params['binwidth'],1000)

    print("2) Generating a map of basis set indices")
    print("\n")
    mymap = mapping(int(params['lmin']), int(params['lmax']), int(params['nbins']), int(params['nlobatto']))
    maparray, Nbas = mymap.gen_map()
    params['Nbas'] = Nbas
    print("Nbas = "+str(Nbas))
    return Nbas,maparray,rgrid

def pot_grid_interp():
    #interpolate potential on the grid
    fl = open(params['working_dir']+params['potential_grid']+".dat",'r')

    esp = []

    for line in fl:
        words = line.split()
        # columns: l, m, i, n, coef

        x = float(words[0])
        y = float(words[1])
        z = float(words[2])
        v = float(words[3])
        esp.append([x,y,z,v])
        
    esp = -1.0 * np.asarray(esp) #NOTE: Psi4 returns ESP for a positive unit charge, so that the potential of a cation is positive. We want it for negaitve unit charge, so we must change sign: attractive interaction

    #plot preparation
    #X = np.linspace(min(esp[:,0]), max(esp[:,0]),100)
    #Y = np.linspace(min(esp[:,1]), max(esp[:,1]),100)
    #X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    
    #fig = plt.figure() 
    #ax = plt.axes(projection="3d")
    #ax.scatter(x, z, v, c='r', marker='o')

    sph_rule = gen_leb_quad(params['scheme'])
    #values = 
    
    start_time = time.time()
    esp_interp = interpolate.LinearNDInterpolator(esp[:,0:3],esp[:,3])
    #esp_interp = interpolate.griddata(esp[:,0:3],esp[:,3],values,method='nearest')
    end_time = time.time()
    #print("Interpolation of " + params['potential_grid'] + " potential took " +  str("%10.3f"%(end_time-start_time)) + "s")
    """
    Z = esp_interp(X, Y, 0)
    plt.pcolormesh(X, Y, Z, shading='auto')
    #plt.plot(esp[:,0], esp[:,1], "o", label="input point")
    plt.colorbar()
    plt.axis("equal")
    plt.show()
    """
    return esp_interp

@jit(nopython=True,parallel=False,fastmath=False) 
def pot_grid_interp_sph(interpolant,r,theta,phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return interpolant(x,y,z)

def gen_leb_quad(scheme):
    "generate lebedev quadrature rules"

    sphgrid = []
    #print("reading Lebedev grid from file:" + "/lebedev_grids/"+str(scheme)+".txt")
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

@jit(nopython=True,parallel=False,fastmath=False) 
def P(l, m, x):
	pmm = np.ones(1,)
	if m>0:
		somx2 = np.sqrt((1.0-x)*(1.0+x))
		fact = 1.0
		for i in range(1,m+1):
			pmm = pmm * (-fact) * somx2
			fact = fact+ 2.0
	
	if l==m :
		return pmm * np.ones(x.shape,)
	
	pmmp1 = x * (2.0*m+1.0) * pmm
	
	if l==m+1:
		return pmmp1
	
	pll = np.zeros(x.shape)
	for ll in range(m+2, l+1):
		pll = ( (2.0*ll-1.0)*x*pmmp1-(ll+m-1.0)*pmm ) / (ll-m)
		pmm = pmmp1
		pmmp1 = pll
	
	return pll


@jit(nopython=True,parallel=False,fastmath=False) 
def divfact(a, b):
	# PBRT style
	if (b == 0):
		return 1.0
	fa = a
	fb = abs(b)
	v = 1.0

	x = fa-fb+1.0
	while x <= fa+fb:
		v *= x;
		x+=1.0

	return 1.0 / v;

LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')
    
@jit(nopython=True,parallel=False,fastmath=False) 
def fast_factorial(n):
    return LOOKUP_TABLE[n]

def factorial(x):
	if(x == 0):
		return 1.0
	return x * factorial(x-1)

@jit(nopython=True,parallel=False,fastmath=False) 
def K(l, m):
	#return np.sqrt((2.0 * l + 1.0) * 0.07957747154594766788 * divfact(l, m))
	return np.sqrt( ((2 * l + 1) * fast_factorial(l-m)) / (4*np.pi*fast_factorial(l+m)) )


@jit(nopython=True,parallel=False,fastmath=False) 
def SH(l, m, theta, phi):
	if m==0 :
		return K(l,m)*P(l,m,np.cos(theta))*np.ones(phi.shape,)
	elif m>0 :
		return np.sqrt(2.0)*K(l,m)*np.cos(m*phi)*P(l,m,np.cos(theta))
	else:
		return np.sqrt(2.0)*K(l,-m)*np.sin(-m*phi)*P(l,-m,np.cos(theta))




@jit(nopython=True,parallel=False,fastmath=False) 
def calc_hmat(N,rgrid,maparray,potmat,G):

    for i in range(N):
        rin = rgrid[maparray[i][2],maparray[i][3]]
        for j in range(i,N):
            if maparray[i][2] == maparray[j][2] and maparray[i][3] == maparray[j][3]:

                w = G[:,2]
                f = np.conjugate(SH( maparray[i][0] , maparray[i][1] , G[:,0], G[:,1] + np.pi  )) *  SH( maparray[j][0] , maparray[j][1] , G[:,0], G[:,1] + np.pi  )
                potmat[i,j] = np.dot(w,f) * 4.0 * np.pi

                potmat[j,i] = potmat[i,j]

    return potmat


params = input.gen_input()

N,maparray,rgrid = prep(params)

hmat = np.zeros((N,N))

quad_schemes = []
quadfilename = params['working_dir']+params['quad_levels_file']
fl = open(quadfilename,'r')

for line in fl:
    words = line.split()
    i = int(words[0])
    n = int(words[1])
    scheme = str(words[2])
    quad_schemes.append([i,n,scheme])


quad_rule = gen_leb_quad(params['scheme'])

#interpolate the ESP:
print("Interpolating electrostatic potential")
#esp_interpolant = pot_grid_interp()

#esp = calc_esp_grid() #calculate esp on quadrature grid



start_time = time.time()
hmat = calc_hmat(N,rgrid,maparray,hmat,quad_rule)    
end_time = time.time()
print("Time for construction of the Hamiltonian: " +  str("%10.3f"%(end_time-start_time)) + "s")


print(hmat)