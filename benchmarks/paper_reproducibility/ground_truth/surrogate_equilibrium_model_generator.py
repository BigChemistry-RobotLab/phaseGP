import numpy as np
from tqdm import tqdm

from solvemassbal_binsearch import solve_massbalance
from sklearn.ensemble import RandomForestClassifier
import joblib
import argparse

# Define a custom argument type for a list of integers
def list_of_floats(arg):
    return list(map(float, arg.split(',')))

def get_pm_concentration_phase(atot, btot):
    c_tot = np.array([atot,btot]).reshape(-1,1)
    Ktable = np.array([K1])
    sigmatable = np.array([sigma1])    

    # solve maas balance for these parameters:
    [c_eq,res_mons,PP,LL] = solve_massbalance(c_tot,sigmatable,Ktable)
    #concentrations = c_eq.ravel()
    return sum(PP[:,0]), max(abs(res_mons))[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_concentration", type=float, default=2e-5)
    parser.add_argument("--grid_size", type=int, default=100)
    parser.add_argument("--Ke", type=float, default=1e6)
    parser.add_argument("--sigma", type=float, default=1e-5)
    parser.add_argument("--fA", type=list_of_floats)
    parser.add_argument("--fB", type=list_of_floats)

    args = parser.parse_args()

    a_tot_list = np.linspace(0.01e-5, args.max_concentration,args.grid_size)
    b_tot_list = np.linspace(0.01e-5, args.max_concentration,args.grid_size)


    #gasconstant = 0.008314472 # in: kJ/K mol
    #T = 333

    fracBB = 0.0

    Ke = args.Ke
    sigma = args.sigma
    print(Ke,sigma)

    for fracAA, fracAB in tqdm(zip(args.fA, args.fB)):

        #RT = gasconstant*T

        KPA_A = Ke*fracAA # add A on top A
        KPB_A = Ke*fracAB # add A on top B
        KPA_B = Ke*fracAB # add B on top A
        KPB_B = Ke*fracBB # add B on top B  

        # parameters for poltype 1 (=P)
        K1 = np.array([[KPA_A, KPB_A], 
                    [KPA_B, KPB_B]])
        sigma1 = np.array([sigma,sigma]).reshape(-1,1)

        # Create meshgrid for all combinations of the two parameters
        A_mesh, B_mesh = np.meshgrid(a_tot_list, b_tot_list)
            
        # Loop through each point and apply the function sequentially
        vectorized_func = np.vectorize(get_pm_concentration_phase,otypes=[float,float])
        sumPP,max_err = vectorized_func(A_mesh, B_mesh)
        

        Z = sumPP > 0.5*(A_mesh+B_mesh)

        X_train = np.column_stack((A_mesh.flatten(), B_mesh.flatten()))
        y_train = Z.flatten()

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        joblib.dump(model, f'ground_truth/surrogate_equilibrium_model/equilibrium_fracAA{fracAA}_fracAB{fracAB}.pkl')
    pass