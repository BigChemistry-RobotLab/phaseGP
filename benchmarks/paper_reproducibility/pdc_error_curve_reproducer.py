import numpy as np
import sklearn
from scipy import stats
import os
import pickle
import argparse
from tqdm import tqdm
seed = 50
use_cross_selection = False
np.random.seed(seed)

from ground_truth.true_phase import true_phase
from ground_truth.flory_phase import flory_true_phase

def swapper(phase_diagram):
    phase_diagram[phase_diagram==2] = 1
    phase_diagram[phase_diagram==1] = 3
    phase_diagram[phase_diagram==0] = 1
    phase_diagram[phase_diagram==3] = 0
    return phase_diagram

def grid_mask(grid):
    mask = grid[:,0] + grid[:,1] < 1
    return grid[mask]

def remove_invalid_points(ax):
    outbounds_x = [0, 1, 1, 0]  # Triangle outbounds fo diagram
    outbounds_y = [1, 0, 1, 1]

    ax.fill(outbounds_x, outbounds_y, color='white')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    return ax

def run_experiments(benchmark_name, total_episodes=20, total_trials=50, initial_points = 10, US_strategy="LC", n_points = 20, ionic_strength=0.75, noise = 0):
    x1 = np.linspace(0, 1, 100)
    x2 = np.linspace(0, 1, 100)

    # Create meshgrid
    X, Y = np.meshgrid(x1, x2)

    # Stack into coordinate pairs - shape will be (n_points, n_points, 2)
    full_grid = np.stack([X, Y], axis=-1).reshape(-1, 2)

    if("equilibrium" in benchmark_name):
        full_grid_labels = true_phase(full_grid*(2e-5), benchmark_name, noise = 0)
    else:
        full_grid_labels = true_phase(full_grid, benchmark_name, noise = 0)

    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)

    # Create meshgrid
    X, Y = np.meshgrid(x, y)

    # Stack into coordinate pairs - shape will be (n_points, n_points, 2)
    grid = np.stack([X, Y], axis=-1).reshape(-1, 2)
    if("equilibrium" in benchmark_name):
        grid_labels = true_phase(grid*(2e-5), benchmark_name, noise = noise)
    else:
        grid_labels = true_phase(grid, benchmark_name, noise = noise)

    all_indices = np.arange(len(grid_labels ))
    error_data = []
    for episode in tqdm(range(total_episodes)):
        
        np.random.seed(episode)
        error_data_list = []

        label_train = np.zeros(grid_labels.shape) - 1
        labeled_indices = np.random.randint(0, len(grid_labels ), size=initial_points)

        
        for trial in range(total_trials+1):
            #print('trial:', trial)
            
            label_train[labeled_indices] = grid_labels[labeled_indices]
                
            lp_model = sklearn.semi_supervised.LabelPropagation(kernel="rbf", gamma = 20)
                
            

            # Get indices NOT in your list
            unlabeled_indices = np.setdiff1d(all_indices, labeled_indices )
            

            lp_model.fit(grid, label_train)
            label_distributions = lp_model.label_distributions_[unlabeled_indices]
            
            
            #print(classification_report(np.copy(grid_list[:,2]), predicted_all_labels))
            preds = lp_model.predict(full_grid)

            correct_predictions = preds != full_grid_labels
            error = np.mean(correct_predictions)

            error_data_list.append(error)
            #print('macro_report', macro_report, classes)
            if US_strategy == 'E':
                pred_entropies = stats.distributions.entropy(label_distributions.T)
                u_score_list = pred_entropies/np.max(pred_entropies)
                uncertainty_index = [unlabeled_indices[np.argmax(pred_entropies)]]
                
                
            elif US_strategy == 'LC':
                u_score_list = 1- np.max(label_distributions, axis = 1)
                uncertainty_index = [unlabeled_indices[np.argmax(1- np.max(label_distributions, axis = 1))]]
                
            elif US_strategy == 'MS':
                u_score_list = []
                for pro_dist in label_distributions:
                    pro_ordered = np.sort(pro_dist)[::-1]
                    margin = pro_ordered[0] - pro_ordered[1]
                    u_score_list.append(margin)

                    uncertainty_index = [unlabeled_indices[np.argmin(u_score_list)]]
            
            
                
            
            labeled_indices = np.concatenate([labeled_indices, np.array(uncertainty_index)])
            

        error_data.append(error_data_list)

    error_data = np.array(error_data)
    return error_data

def run_experiments_condensate(benchmark_name, total_episodes=20, total_trials=50, initial_points = 10, US_strategy="LC", n_points = 20, ionic_strength=0.75, noise = 0):
    x1 = np.linspace(0.001, 1, 100)
    x2 = np.linspace(0.001, 1, 100)

    # Create meshgrid
    X, Y = np.meshgrid(x1, x2)


    # Stack into coordinate pairs - shape will be (n_points, n_points, 2)
    full_grid = np.stack([X, Y], axis=-1).reshape(-1, 2)


    sum = np.sum(full_grid, axis=1)
    full_grid  = full_grid[sum <= 0.999]

    full_grid_labels = flory_true_phase(full_grid, ionic_strength=ionic_strength)
    full_grid_labels = np.array(full_grid_labels)
    x = np.linspace(0.001, 1, n_points)
    y = np.linspace(0.001, 1, n_points)

    # Create meshgrid
    X, Y = np.meshgrid(x, y)

    # Stack into coordinate pairs - shape will be (n_points, n_points, 2)
    grid = np.stack([X, Y], axis=-1).reshape(-1, 2)
    sum = np.sum(grid, axis=1)
    grid = grid[sum <= 0.999]
    grid_labels = flory_true_phase(grid, ionic_strength=ionic_strength)
    grid_labels = np.array(grid_labels)
    

    all_indices = np.arange(len(grid_labels ))
    error_data = []
    for episode in tqdm(range(total_episodes)):
        np.random.seed(episode)
        error_data_list = []

        label_train = np.zeros(grid_labels.shape) - 1
        labeled_indices = np.random.randint(0, len(grid_labels ), size=initial_points)

        
        for trial in range(total_trials+1):
            #print('trial:', trial)
            
            label_train[labeled_indices] = grid_labels[labeled_indices]
                
            lp_model = sklearn.semi_supervised.LabelPropagation(kernel="rbf", gamma = 20)
                
            

            # Get indices NOT in your list
            unlabeled_indices = np.setdiff1d(all_indices, labeled_indices )
            

            lp_model.fit(grid, label_train)
            label_distributions = lp_model.label_distributions_[unlabeled_indices]
            
            
            #print(classification_report(np.copy(grid_list[:,2]), predicted_all_labels))
            preds = lp_model.predict(full_grid)

            correct_predictions = preds != full_grid_labels
            error = np.mean(correct_predictions)

            error_data_list.append(error)
            #print('macro_report', macro_report, classes)
            if US_strategy == 'E':
                pred_entropies = stats.distributions.entropy(label_distributions.T)
                u_score_list = pred_entropies/np.max(pred_entropies)
                uncertainty_index = [unlabeled_indices[np.argmax(pred_entropies)]]
                
                
            elif US_strategy == 'LC':
                u_score_list = 1- np.max(label_distributions, axis = 1)
                uncertainty_index = [unlabeled_indices[np.argmax(1- np.max(label_distributions, axis = 1))]]
                
            elif US_strategy == 'MS':
                u_score_list = []
                for pro_dist in label_distributions:
                    pro_ordered = np.sort(pro_dist)[::-1]
                    margin = pro_ordered[0] - pro_ordered[1]
                    u_score_list.append(margin)

                    uncertainty_index = [unlabeled_indices[np.argmin(u_score_list)]]

            labeled_indices = np.concatenate([labeled_indices, np.array(uncertainty_index)])
            

        error_data.append(error_data_list)

    error_data = np.array(error_data)
    return error_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", help="Name of the benchmark to reproduce", type=str)
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--candidate_size", type=int, default=20)
    parser.add_argument("--n_sampled_points", type=int, default=100)
    parser.add_argument("--n_initial_points", type=int, default=10)
    args = parser.parse_args()

    print(f"CALCULATING ERROR CURVES FOR BENCHMARK={args.benchmark}")
    #This correspond vanilla, Single-Source PhaseTransfer and Multi-Source PhaseTransfer respectively
    n_initial_points = args.n_initial_points
    candidate_grid_size = 30 #For flory-huggins only
    noise = 0
    
    benchmark_name_dictionary = {"biological_condensate": "flory_low_to_high",
                                 "supramolecular_copolymerization":"equilibrium_high_correlation",
                                 "batch_biological_condensate": "batch_flory_low_to_high",
                                 "batch_supramolecular_copolymerization":"batch_equilibrium_high_correlation"}
    original_benchmark_name = args.benchmark

    if(args.benchmark in benchmark_name_dictionary.keys()):
        args.benchmark = benchmark_name_dictionary[args.benchmark]

    if("flory" not in args.benchmark):
        #Selector of source and target benchmarks
        if(args.benchmark == "diagonal"):
            benchmark_name = "diagonal_circle"
            n_samples_per_it = 1
        elif(args.benchmark == "sine"):
            benchmark_name = "sine_wave_circle"
            n_samples_per_it = 1
        elif(args.benchmark == "noisy_sine"):
            benchmark_name = "sine_wave_circle"
            n_samples_per_it = 1
            noise = 0.1
        elif(args.benchmark == "equilibrium_high_correlation"):
            benchmark_name = "equilibrium_fracAA0.01_fracAB5"
            n_samples_per_it = 1
        elif(args.benchmark == "equilibrium_low_correlation"):
            benchmark_name = "equilibrium_fracAA0.01_fracAB5"
            n_samples_per_it = 1
        elif(args.benchmark == "equilibrium_low_correlation_diagonal"):
            benchmark_name = "equilibrium_fracAA0.01_fracAB5"
            n_samples_per_it = 1

        else:
            raise Exception(f"Wrong benchmark name: {args.benchmark}")
        

        error_data = run_experiments(benchmark_name, total_episodes=args.n_seeds, total_trials=args.n_sampled_points, initial_points = args.n_initial_points, noise = noise)
            
           
    else:
        if(args.benchmark == "flory_high_to_low"):
            ionic_strength = 0.01
            n_iterations = 100
            n_samples_per_it = 1
        elif(args.benchmark == "flory_low_to_high"):
            ionic_strength = 0.75
            n_iterations = 100
            n_samples_per_it = 1
        else:
            raise Exception("Wrong benchmark name")
        
        
        error_data = run_experiments_condensate(args.benchmark, total_episodes=args.n_seeds, total_trials=args.n_sampled_points, initial_points = args.n_initial_points, noise = noise, ionic_strength=ionic_strength)
        

    os.makedirs('synthetic_error_rate_data/pdc/', exist_ok=True)
    with open('synthetic_error_rate_data/pdc/'+original_benchmark_name+'.pkl', 'wb') as file: 
            pickle.dump(error_data, file)
