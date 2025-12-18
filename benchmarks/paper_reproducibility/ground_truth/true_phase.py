import numpy as np
import torch
import joblib

# =============================================================================
# True Phase Function
# =============================================================================

def true_phase(x, benchmark_name, center = np.array([0.25, 0.25]), radius = 0.2, noise = 0):

    if isinstance(x, torch.Tensor):
        if(len(x.shape)==1):
            x = x.reshape(1,-1)
        noisy_signal = torch.rand_like(x) * 2*noise - noise #Added noise to inputs
        x = x + noisy_signal
        if(benchmark_name == "diagonal"):
            below_diagonal = (x[:, 0] + x[:, 1] <= 1.0) #usual diagonal
            phases = below_diagonal
        elif(benchmark_name == "diagonal_plus_offset"):
            below_diagonal = (x[:, 0] + x[:, 1] <= 1.2) #usual diagonal
            phases = below_diagonal
        elif(benchmark_name == "equilibrium_diagonal"):
            below_diagonal = (x[:, 0] + x[:, 1] <= 1.0*2e-5) #usual diagonal
            phases = below_diagonal
        elif(benchmark_name == "equilibrium_diagonal_plus_offset"):
            below_diagonal = (x[:, 0] + x[:, 1] <= 1.2*2e-5) #usual diagonal
            phases = below_diagonal
        elif(benchmark_name == "diagonal_minus_offset"):
            below_diagonal = (x[:, 0] + x[:, 1] <= 0.8) #usual diagonal
            phases = below_diagonal
        elif(benchmark_name == "diagonal_circle"):
            below_diagonal = (x[:, 0] + x[:, 1] <= 1.0)
            distances = torch.sqrt(torch.sum((x - torch.tensor(center, dtype=x.dtype))**2, dim=1))
            inside_circle = (distances <= radius)
            phases = below_diagonal ^ inside_circle
        elif(benchmark_name == "circle"):
            distances = torch.sqrt(torch.sum((x - torch.tensor(center, dtype=x.dtype))**2, dim=1))
            inside_circle = (distances <= radius)
            phases = inside_circle
        elif(benchmark_name == "diagonal_rotated_circle"):
            below_diagonal = (x[:, 0] + (5/3)*x[:, 1] <= 4/3)
            distances = torch.sqrt(torch.sum((x - torch.tensor(center, dtype=x.dtype))**2, dim=1))
            inside_circle = (distances <= radius)
            phases = below_diagonal ^ inside_circle
        elif(benchmark_name == "diagonal_small_plus_offset_circle"):
            below_diagonal = (x[:, 0] + x[:, 1] <= 1.1)
            distances = torch.sqrt(torch.sum((x - torch.tensor(center, dtype=x.dtype))**2, dim=1))
            inside_circle = (distances <= radius)
            phases = below_diagonal ^ inside_circle
        elif(benchmark_name == "diagonal_big_plus_offset_circle"):
            below_diagonal = (x[:, 0] + x[:, 1] <= 1.3)
            distances = torch.sqrt(torch.sum((x - torch.tensor(center, dtype=x.dtype))**2, dim=1))
            inside_circle = (distances <= radius)
            phases = below_diagonal ^ inside_circle
        elif(benchmark_name == "diagonal_plus_offset_circle"):
            below_diagonal = (x[:, 0] + x[:, 1] <= 1.2)
            distances = torch.sqrt(torch.sum((x - torch.tensor(center, dtype=x.dtype))**2, dim=1))
            inside_circle = (distances <= radius)
            phases = below_diagonal ^ inside_circle
        elif("sine_wave_offset" in benchmark_name):
            offset = benchmark_name.split("_")[-1]
            offset = float(offset)

            theta = -np.pi/4
            rotation = np.array([[np.cos(theta) , np.sin(theta)],
                           [-np.sin(theta), np.cos(theta)]])
        
            coordinates = x.clone()
            coordinates[:, 1] = coordinates[:, 1] - 0.9
            coordinates =  coordinates @ rotation.T
            below_sine = (torch.sin(5*torch.pi*(coordinates[:, 0] + offset*torch.pi))+1)/10 >= coordinates[:, 1]

            phases = below_sine
        elif(benchmark_name == "sine_wave_circle"):
            theta = -np.pi/4
            rotation = np.array([[np.cos(theta) , np.sin(theta)],
                           [-np.sin(theta), np.cos(theta)]])
        
            coordinates = x.clone()
            coordinates[:, 1] = coordinates[:, 1] - 0.9
            coordinates =  coordinates @ rotation.T

            below_sine = (torch.sin((5/1.1)*torch.pi*coordinates[:, 0])+1)/10 >= coordinates[:, 1] #Change in amplitude
            distances = torch.sqrt(torch.sum((x - torch.tensor(center, dtype=x.dtype))**2, dim=1))
            inside_circle = (distances <= radius)
            phases = below_sine ^ inside_circle
        elif("equilibrium" in benchmark_name):
            #x =x*2e-5
            numpy_copy = x.numpy().copy()
            model = joblib.load('ground_truth/surrogate_equilibrium_model/'+benchmark_name+'.pkl')
            if(len(numpy_copy.shape)==1):
                numpy_copy = numpy_copy.reshape(1,-1)
            phases = model.predict(numpy_copy)
            del numpy_copy
            phases = torch.from_numpy(phases)
        else:
            raise Exception(f"Wrong True Phase Name: {benchmark_name}")
        
        flip_mask = torch.bernoulli(torch.full_like(phases, noise/2, dtype=torch.float)).bool() #Flip noise
        phases[flip_mask] = ~phases[flip_mask]
        return phases.float()
    else:
        noisy_signal = np.random.rand(*x.shape) * 2*noise - noise #added noise to input
        x = x + noisy_signal
        if(benchmark_name == "diagonal"):
            below_diagonal = (x[:, 0] + x[:, 1] <= 1.0) #usual diagonal
            phases = below_diagonal
        elif(benchmark_name == "diagonal_plus_offset"):
            below_diagonal = (x[:, 0] + x[:, 1] <= 1.2) #usual diagonal
            phases = below_diagonal
        elif(benchmark_name == "diagonal_minus_offset"):
            below_diagonal = (x[:, 0] + x[:, 1] <= 0.8) #usual diagonal
            phases = below_diagonal
        elif(benchmark_name == "diagonal_circle"):
            below_diagonal = (x[:, 0] + x[:, 1] <= 1.0)
            distances = np.sqrt(np.sum((x - center)**2, axis=1))
            inside_circle = (distances <= radius)
            phases = np.logical_xor(below_diagonal, inside_circle)
        elif(benchmark_name == "circle"):
            distances = np.sqrt(np.sum((x - center)**2, axis=1))
            inside_circle = (distances <= radius)
            phases = inside_circle
        elif(benchmark_name == "diagonal_rotated_circle"):
            below_diagonal = (x[:, 0] + (5/3)*x[:, 1] <= 4/3)
            distances = np.sqrt(np.sum((x - center)**2, axis=1))
            inside_circle = (distances <= radius)
            phases = np.logical_xor(below_diagonal, inside_circle)
        elif(benchmark_name == "diagonal_small_plus_offset_circle"):
            below_diagonal = (x[:, 0] + x[:, 1] <= 1.1)
            distances = np.sqrt(np.sum((x - center)**2, axis=1))
            inside_circle = (distances <= radius)
            phases = np.logical_xor(below_diagonal, inside_circle)
        elif(benchmark_name == "diagonal_big_plus_offset_circle"):
            below_diagonal = (x[:, 0] + x[:, 1] <= 1.3)
            distances = np.sqrt(np.sum((x - center)**2, axis=1))
            inside_circle = (distances <= radius)
            phases = np.logical_xor(below_diagonal, inside_circle)
        elif(benchmark_name == "diagonal_plus_offset_circle"):
            below_diagonal = (x[:, 0] + x[:, 1] <= 1.2)
            distances = np.sqrt(np.sum((x - center)**2, axis=1))
            inside_circle = (distances <= radius)
            phases = np.logical_xor(below_diagonal, inside_circle)
        elif("sine_wave_offset" in benchmark_name):
            offset = benchmark_name.split("_")[-1]
            offset = float(offset)

            theta = -np.pi/4
            rotation = np.array([[np.cos(theta) , np.sin(theta)],
                           [-np.sin(theta), np.cos(theta)]])
        
            coordinates = np.copy(x)
            coordinates[:, 1] = coordinates[:, 1] - 0.9
            coordinates =  coordinates @ rotation.T

            below_sine = (np.sin(5*np.pi*(coordinates[:, 0]+ offset*np.pi))+1)/10 >= coordinates[:, 1]
            phases = below_sine
        elif(benchmark_name == "sine_wave_circle"):
            theta = -np.pi/4
            rotation = np.array([[np.cos(theta) , np.sin(theta)],
                           [-np.sin(theta), np.cos(theta)]])
            
            coordinates = np.copy(x)
            coordinates[:, 1] = coordinates[:, 1] - 0.9
            coordinates =  coordinates @ rotation.T

            below_sine = (np.sin((5/1.1)*np.pi*coordinates[:, 0])+1)/10 >= coordinates[:, 1]
            distances = np.sqrt(np.sum((x - center)**2, axis=1))
            inside_circle = (distances <= radius)
            phases = np.logical_xor(below_sine, inside_circle)
        elif("equilibrium" in benchmark_name):
            #x =x*2e-5
            model = joblib.load('ground_truth/surrogate_equilibrium_model/'+benchmark_name+'.pkl')
            phases = model.predict(x)
        else:
            raise Exception(f"Wrong True Phase Name: {benchmark_name}")
        
        flip_mask = np.random.random(phases.shape) < noise/2          #Flip noise
        phases[flip_mask] = np.logical_not(phases[flip_mask])
        return phases.astype(float)