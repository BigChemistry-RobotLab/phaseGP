import numpy as np
import torch
from phaseGP.utils import ensure_numpy

class ReentrantFloryHuggins:
    def __init__(self, ionic_strength):
        self.N_P = 200          # Protein chain length
        self.N_R = 100          # RNA chain length  
        self.N_S = 1            # Salt

        # Salt-Dependent Interaction Parameters
        self.chi_PR_0 = -1.8     # Non-electrostatic baseline interaction
        self.chi_PR_A = 20
        self.chi_PR_B = 30
        self.chi_PR_alpha = 6
        self.chi_PR_elec = -0.4 # Electrostatic attraction (negative = attractive)
        self.k = 0.329          # Debye-Hückel parameter (nm⁻¹·M⁻¹/²) at 25°C
        self.a = 1.0            # Effective interaction distance (nm)
        self.chi_PS_0 = 0.1     # Baseline protein-salt interaction
        self.k_PS = 0.3         # Salting-out coefficient (M⁻¹)
        self.chi_RS_0 = -0.2     # Baseline RNA-salt interaction
        self.k_RS1 = 0.6         # RNA salting-out coefficient (M⁻¹)
        self.k_RS2 = -0.8         # RNA salting-out coefficient (M⁻¹)
        self.k_RS = 0.3      

        self.I = ionic_strength

    def chi_parameters(self, phi_P, phi_R,):
        """Calculate interaction parameters at given ionic strength"""
        
        # Protein-RNA: Electrostatic with Debye screening
        chi_PR = (self.chi_PR_0 + self.chi_PR_A *(phi_P + phi_R)**2-self.chi_PR_B*(phi_P + phi_R)**4)*np.exp(-self.chi_PR_alpha*self.I)

        
        # Protein-Salt Interaction (Salting-out)
        chi_PS = self.chi_PS_0 + self.k_PS * self.I

        # RNA-Salt Interaction (Strong Salting-out) CHANGE TO NON MONOTONIC
        #chi_RS = self.chi_RS_0 + self.k_RS * I
        chi_RS = self.chi_RS_0 + self.k_RS1 * self.I - self.k_RS2 * self.I**2

        return chi_PR, chi_PS, chi_RS
    
    def free_energy_density(self, phi_P, phi_R):
        """Calculate Flory-Huggins free energy density"""
        phi_S = 1 - phi_P - phi_R
        if(phi_S < 0 or phi_P < 0 or phi_R < 0):
            return np.nan
        
        chi_PR, chi_PS, chi_RS = self.chi_parameters(phi_P, phi_R)
        # Entropy terms (avoid numerical problems when phi close to 0)
        entropy = 0
        if phi_P > 1e-12:
            entropy += (phi_P / self.N_P) * np.log(phi_P)
        if phi_R > 1e-12:
            entropy += (phi_R / self.N_R) * np.log(phi_R)
        if phi_S > 1e-12:
            entropy += (phi_S / self.N_S) * np.log(phi_S)
        
        # Interaction terms
        interaction = (chi_PR * phi_P * phi_R + 
                      chi_PS * phi_P * phi_S + 
                      chi_RS * phi_R * phi_S)
        
        return entropy + interaction
    
    def check_stability(self, x):
        """Check if composition is thermodynamically stable"""
        

        phi_P, phi_R = x[:,0], x[:,1]
        phi_S = 1 - phi_P - phi_R
        #if(phi_S < 0 or phi_P < 0 or phi_R < 0):
        #    return np.nan
        chi_PR, chi_PS, chi_RS = self.chi_parameters(phi_P, phi_R)
        # Calculate Hessian matrix
        H = np.zeros((len(x), 2, 2))  # 2x2 for independent components
        
        H[:, 0,0] = 1/(self.N_P * phi_P) + 1/(self.N_S * phi_S) - 2*chi_PS
        H[:, 1,1] = 1/(self.N_R * phi_R) + 1/(self.N_S * phi_S) - 2*chi_RS
        H[:, 0,1] = H[:, 1,0] = chi_PR - chi_PS - chi_RS + 1/(self.N_S * phi_S)
        
        # Stability criterion: all eigenvalues > 0
        eigenvals = np.linalg.eigvals(H)
        return np.all(eigenvals > 0, axis = 1)

        
    def check_single_stability(self, x):
        """Check if composition is thermodynamically stable"""
        phi_P , phi_R = x[0,0], x[0,1]
        phi_S = 1 - phi_P - phi_R
        # Calculate Hessian matrix
        chi_PR, chi_PS, chi_RS = self.chi_parameters(phi_P, phi_R)
        H = np.zeros((2, 2))  # 2x2 for independent components
        
        H[0,0] = 1/(self.N_P * phi_P) + 1/(self.N_S * phi_S) - 2*chi_PS
        H[1,1] = 1/(self.N_R * phi_R) + 1/(self.N_S * phi_S) - 2*chi_RS
        H[0,1] = H[1,0] = chi_PR - chi_PS - chi_RS + 1/(self.N_S * phi_S)
        
        # Stability criterion: all eigenvalues > 0
        eigenvals = np.linalg.eigvals(H)
        #print(H)
        if np.all(eigenvals > 0):
            return torch.Tensor([1])
        else:
            return torch.Tensor([0])
        
def flory_true_phase(x, ionic_strength):

    x = ensure_numpy(x)
    simulation = ReentrantFloryHuggins(ionic_strength=ionic_strength)
    result = np.zeros(len(x)) + 2

    if(len(x) > 1):
        mask = x[:,0] + x[:, 1] < 1
        result[mask] = simulation.check_stability(x[mask])
        return torch.from_numpy(result)
    else:
        result = simulation.check_single_stability(x)
        return result