
import numpy as np


def sph2cart(r,theta,phi):
    x=r*np.sin(theta)*np.cos(phi)
    y=r*np.sin(theta)*np.sin(phi)
    z=r*np.cos(theta)
    return x,y,z



def calc_repulsion_energy():

    au_to_ev    = 27.211 
    r_array     = np.array([4.0, 2.0, 2.0, 4.0, 0.0 ])
    theta_array = np.array([0.0, np.pi/2.0, np.pi/2.0, 2.19, 0.0])
    phi_array   = np.array([0.0, 0.0, np.pi/2.0, 3.93, 0.0])
    Q_array     = np.array([0.5, 1.5, 1.0, 0.25, 2.0])


    Nnuc = r_array.shape[0]
    Vnn = 0.0
    for inuc in range(Nnuc):
        for jnuc in range(inuc+1,Nnuc):
            Vnn += Q_array[inuc] * Q_array[jnuc] * (r_array[inuc]**2 + r_array[jnuc]**2 - 2.0 * r_array[inuc] * r_array[jnuc] *\
                ( np.sin(theta_array[inuc]) * np.sin(theta_array[jnuc]) * np.cos(phi_array[inuc]-phi_array[jnuc])+\
                    np.cos(theta_array[inuc]) * np.cos(theta_array[jnuc]) ))**(-0.5)

    print("Nuclear repulsion energy Vnn = " + str(Vnn) + " a.u.")
    print("Nuclear repulsion energy Vnn = " + str(Vnn*au_to_ev) + " ev")


if __name__ == "__main__":   



    calc_repulsion_energy()