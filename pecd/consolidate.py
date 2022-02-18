import os
import sys
import json


class Avobs:
    def __init__(self,params):
        self.params = params
        self.helicity = params['helicity']
        
    def calc_bcoeffs(self):
        pass

    def plot_bcoeffs(self):
        pass
    
    
    """


    Wav = np.zeros((grid_theta.shape[0],grid_r.shape[0]), dtype = float)




    if params['density_averaging'] == True:
        WDMATS  = gen_wigner_dmats( N_Euler, 
                                    params['Jmax'], 
                                    grid_euler)

        #calculate rotational density at grid (alpha, beta, gamma) = (n_grid_euler, 3)
        grid_rho, rho = ROTDENS.calc_rotdens(   grid_euler,
                                                WDMATS,
                                                params) 


          
        print(n_grid_euler_2d)
        grid_rho, rho = ROTDENS.calc_rotdens( grid_euler_2d,
                                    WDMATS,
                                    params) 
        print("shape of rho")
        print(np.shape(rho))
        #print(rho.shape)
        #PLOTS.plot_rotdens(rho[:].real, grid_euler_2d)
        """
if __name__ == "__main__":   

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    print(" ")
    print("---------------------- START CONSOLIDATE --------------------")
    print(" ")
    
    os.chdir(sys.argv[1])
    path = os.getcwd()

    # read input_propagate
    with open('input_prop', 'r') as input_file:
        params_prop = json.load(input_file)

    #read input_analyze
    with open('input_analyze', 'r') as input_file:
        params_analyze = json.load(input_file)

    #check compatibility
    #check_compatibility(params_prop,params_analyze)

    #combine inputs
    params = {}
    params.update(params_prop)
    params.update(params_analyze)

    Obs = Avobs(params)