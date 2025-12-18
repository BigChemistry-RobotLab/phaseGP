import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import os

from phaseGP.models import PhaseGP, PhaseTransferGP
from phaseGP.utils import brute_sample_new_points, get_grid, set_seeds

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

def source_active_learning_loop(benchmark_name, initial_sample_points,  n_iterations=50, candidate_size = 30, test_size = 100, n_samples_per_it=1, noise=0):
    #train_x = torch.rand(n_initial_points, 2)
    train_x = initial_sample_points
    train_y = true_phase(train_x, benchmark_name=benchmark_name ,noise=noise)

    if ("equilibrium" in benchmark_name):
        min_scale = 0
        max_scale = 2e-5
    else:
        min_scale = 0
        max_scale = 1

    candidate_grid = get_grid(min_scale,max_scale, candidate_size)
    test_grid = get_grid(min_scale,max_scale, test_size)
    test_y = true_phase(test_grid, benchmark_name=benchmark_name )

    error_curve = []

    for i in range(n_iterations+1):
        model = PhaseGP(train_x, min_scale=min_scale, max_scale=max_scale )
        model.fit(train_x, train_y)

        new_point, new_index_list = brute_sample_new_points(model, candidate_grid, train_x,n_sample=n_samples_per_it ,return_index=True)
        mask = torch.ones(len(candidate_grid), dtype=torch.bool)
        mask[new_index_list] = False
        candidate_grid = candidate_grid[mask]
        #candidate_grid = torch.cat([candidate_grid[:new_index], candidate_grid[new_index + 1:]]) #remove the chosen index from candidate grid

        new_point_y = true_phase(new_point, benchmark_name=benchmark_name,noise=noise)
        train_x = torch.cat([train_x, new_point], dim=0)
        train_y = torch.cat([train_y, new_point_y], dim=0)

        pred_points = model.predict(test_grid)
        corect_predictions = pred_points != test_y
        error = torch.mean(corect_predictions.float())
        error_curve.append(error)
    return model, error_curve, train_x

def tl_active_learning_loop(source_model_list, benchmark_name, initial_sample_points, n_iterations=50, n_samples_per_it=1,
                             candidate_size = 30, test_size = 100, prior_aggregation="linear", noise = 0):
    train_x = initial_sample_points
    train_y = true_phase(train_x, benchmark_name=benchmark_name, noise=noise )

    if ("equilibrium" in benchmark_name):
        min_scale = 0
        max_scale = 2e-5
    else:
        min_scale = 0
        max_scale = 1

    candidate_grid = get_grid(min_scale,max_scale, candidate_size)
    test_grid = get_grid(min_scale,max_scale, test_size)
    test_y = true_phase(test_grid, benchmark_name=benchmark_name )

    error_curve = []

    for i in range(n_iterations+1):
        model = PhaseTransferGP(source_model_list, train_x, min_scale=min_scale, max_scale=max_scale, prior_aggregation=prior_aggregation)
        model.fit(train_x, train_y)

        new_point, new_index_list = brute_sample_new_points(model, candidate_grid, train_x,n_sample=n_samples_per_it ,return_index=True)
        mask = torch.ones(len(candidate_grid), dtype=torch.bool)
        mask[new_index_list] = False
        candidate_grid = candidate_grid[mask]

        new_point_y = true_phase(new_point, benchmark_name=benchmark_name,noise=noise)
        train_x = torch.cat([train_x, new_point], dim=0)
        train_y = torch.cat([train_y, new_point_y], dim=0)

        pred_points = model.predict(test_grid)

        corect_predictions = pred_points != test_y
        error = torch.mean(corect_predictions.float())
        error_curve.append(error)
    return model, error_curve, train_x

def flory_source_active_learning_loop(ionic_strength, initial_sample_points,n_iterations=50, candidate_size = 30, test_size = 100, n_samples_per_it=1):
    candidate_grid = get_grid(0.001,0.999, candidate_size)
    candidate_grid = grid_mask(candidate_grid)

    train_x = initial_sample_points
    train_y = flory_true_phase(train_x, ionic_strength=ionic_strength )


    test_grid = get_grid(0.001,0.999, test_size)
    test_grid = grid_mask(test_grid)
    test_y = flory_true_phase(test_grid, ionic_strength=ionic_strength )

    error_curve = []

    for i in range(n_iterations+1):
        model = PhaseGP(train_x, min_scale=0, max_scale=1)
        model.fit(train_x, train_y)

        new_point, new_index_list = brute_sample_new_points(model, candidate_grid, train_x,n_sample=n_samples_per_it ,return_index=True)
        mask = torch.ones(len(candidate_grid), dtype=torch.bool)
        mask[new_index_list] = False
        candidate_grid = candidate_grid[mask]

        new_point_y = flory_true_phase(new_point, ionic_strength=ionic_strength )
        train_x = torch.cat([train_x, new_point], dim=0)
        train_y = torch.cat([train_y, new_point_y], dim=0)

        pred_points = model.predict(test_grid)
        corect_predictions = pred_points != test_y
        error = torch.mean(corect_predictions.float())
        error_curve.append(error)
    return model, error_curve, train_x

def flory_tl_active_learning_loop(source_model_list, ionic_strength,initial_sample_points, n_iterations=50, n_samples_per_it=1,
                             candidate_size = 30, test_size = 100, prior_aggregation="linear"):
    candidate_grid = get_grid(0.001,0.999, candidate_size)
    candidate_grid = grid_mask(candidate_grid)

    train_x = initial_sample_points
    train_y = flory_true_phase(train_x, ionic_strength=ionic_strength )


    test_grid = get_grid(0.001,0.999, test_size)
    test_grid = grid_mask(test_grid)
    test_y = flory_true_phase(test_grid, ionic_strength=ionic_strength )

    error_curve = []

    for i in range(n_iterations+1):
        model = PhaseTransferGP(source_model_list, train_x, min_scale=0, max_scale=1, prior_aggregation=prior_aggregation)
        model.fit(train_x, train_y)

        new_point, new_index_list = brute_sample_new_points(model, candidate_grid, train_x,n_sample=n_samples_per_it ,return_index=True)
        mask = torch.ones(len(candidate_grid), dtype=torch.bool)
        mask[new_index_list] = False
        candidate_grid = candidate_grid[mask]

        new_point_y = flory_true_phase(new_point, ionic_strength=ionic_strength )
        train_x = torch.cat([train_x, new_point], dim=0)
        train_y = torch.cat([train_y, new_point_y], dim=0)

        pred_points = model.predict(test_grid)

        corect_predictions = pred_points != test_y
        error = torch.mean(corect_predictions.float())
        error_curve.append(error)
    return model, error_curve, train_x


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
    error_curve_dict = {"vanilla": [], "tl_single": [], "tl_linear": []}
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
            sources = ["diagonal", "diagonal_plus_offset", "diagonal_minus_offset"]
            benchmark_name = "diagonal_circle"
            n_samples_per_it = 1
        elif(args.benchmark == "sine"):
            sources = ["sine_wave_offset_0", "sine_wave_offset_1", "sine_wave_offset_2"]
            benchmark_name = "sine_wave_circle"
            n_samples_per_it = 1
        elif(args.benchmark == "noisy_sine"):
            sources = ["sine_wave_offset_0", "sine_wave_offset_1", "sine_wave_offset_2"]
            benchmark_name = "sine_wave_circle"
            n_samples_per_it = 1
            noise = 0.1
        elif(args.benchmark == "equilibrium_high_correlation"):
            sources = ["equilibrium_fracAA0.01_fracAB0.5", "equilibrium_fracAA0.1_fracAB5"]
            #sources = ["equilibrium_fracAA1_fracAB1", "equilibrium_fracAA0.1_fracAB0.5"]
            benchmark_name = "equilibrium_fracAA0.01_fracAB5"
            n_samples_per_it = 1
        elif(args.benchmark == "equilibrium_low_correlation"):
            sources = ["equilibrium_fracAA1_fracAB1", "equilibrium_fracAA0.1_fracAB0.5"]
            benchmark_name = "equilibrium_fracAA0.01_fracAB5"
            n_samples_per_it = 1
        elif(args.benchmark == "equilibrium_low_correlation_diagonal"):
            sources = ["diagonal", "equilibrium_fracAA1_fracAB1", "equilibrium_fracAA0.1_fracAB0.5"]
            benchmark_name = "equilibrium_fracAA0.01_fracAB5"
            n_samples_per_it = 1

        elif(args.benchmark == "batch_diagonal"):
            sources = ["diagonal", "diagonal_plus_offset", "diagonal_minus_offset"]
            benchmark_name = "diagonal_circle"
            n_samples_per_it = 10
        elif(args.benchmark == "batch_sine"):
            sources = ["sine_wave_offset_0", "sine_wave_offset_1", "sine_wave_offset_2"]
            benchmark_name = "sine_wave_circle"
            n_samples_per_it = 10
        elif(args.benchmark == "batch_noisy_sine"):
            sources = ["sine_wave_offset_0", "sine_wave_offset_1", "sine_wave_offset_2"]
            benchmark_name = "sine_wave_circle"
            n_samples_per_it = 10
        elif(args.benchmark == "batch_equilibrium_high_correlation"):
            sources = ["equilibrium_fracAA0.01_fracAB0.5", "equilibrium_fracAA0.1_fracAB5"]
            benchmark_name = "equilibrium_fracAA0.01_fracAB5"
            n_samples_per_it = 10
        elif(args.benchmark == "batch_equilibrium_low_correlation"):
            sources = ["equilibrium_fracAA1_fracAB1", "equilibrium_fracAA0.1_fracAB0.5"]
            benchmark_name = "equilibrium_fracAA0.01_fracAB5"
            n_samples_per_it = 10
        elif(args.benchmark == "batch_equilibrium_low_correlation_diagonal"):
            sources = ["diagonal", "equilibrium_fracAA1_fracAB1", "equilibrium_fracAA0.1_fracAB0.5"]
            benchmark_name = "equilibrium_fracAA0.01_fracAB5"
            n_samples_per_it = 10
        else:
            raise Exception("Wrong benchmark name")
        
        n_iterations = int(args.n_sampled_points/n_samples_per_it)
        for i in tqdm(range(args.n_seeds)):
            set_seeds(i)
            source_model_list = []

            #Set up initial points
            initial_sample_points = torch.rand(n_initial_points, 2)
            print("Source Models")
            print("--------------------------")
            for source_benchmark_name in sources:
                if("equilibrium" in source_benchmark_name):
                    source_initial_sample_points = initial_sample_points*2e-5
                else:
                    source_initial_sample_points = initial_sample_points
                model, _, _ = source_active_learning_loop(source_benchmark_name, initial_sample_points =source_initial_sample_points,
                                                           n_iterations=int(n_iterations/2), n_samples_per_it=n_samples_per_it,candidate_size=args.candidate_size,noise=noise)
                source_model_list.append(model)
            if("equilibrium" in benchmark_name):
                initial_sample_points = initial_sample_points*2e-5
            print("Vanilla Model")
            print("--------------------------")
            #Save vanilla error curve
            vanilla_model, vanilla_error_curve, vanilla_sampled_points = source_active_learning_loop(benchmark_name,initial_sample_points =initial_sample_points,
                                                                                                      n_iterations=n_iterations, n_samples_per_it=n_samples_per_it,
                                                                                                      candidate_size=args.candidate_size, noise=noise)
            error_curve_dict["vanilla"].append(vanilla_error_curve)

            print("Single-Source PhaseTransfer")
            print("--------------------------")
            #Save single TL error curve
            tl_single_model, tl_single_error_curve, tl_single_sampled_points = tl_active_learning_loop([source_model_list[0]], benchmark_name,initial_sample_points =initial_sample_points,
                                                                                                        n_iterations=n_iterations, n_samples_per_it=n_samples_per_it,
                                                                                                        candidate_size=args.candidate_size, noise=noise)
            error_curve_dict["tl_single"].append(tl_single_error_curve)

            print("Multi-Source PhaseTransfer")
            print("--------------------------")
            #Save multi linear TL error curve
            tl_linear_model, tl_linear_error_curve, tl_linear_sampled_points = tl_active_learning_loop(source_model_list, benchmark_name, initial_sample_points =initial_sample_points,
                                                                                                        prior_aggregation="linear", n_iterations=n_iterations, n_samples_per_it=n_samples_per_it,
                                                                                                        candidate_size=args.candidate_size, noise=noise)
            error_curve_dict["tl_linear"].append(tl_linear_error_curve)

    else:
        if(args.benchmark == "flory_high_to_low"):
            #High to low source -> [0.15, 0.3, 0.45] | target -> 0.01
            source_ionic_strengths = [0.15, 0.3, 0.45]
            target_ionic_strength = 0.01
            n_iterations = 100
            n_samples_per_it = 1
        elif(args.benchmark == "flory_low_to_high"):
            #Low to high source -> [0.6, 0.45, 0.3] | target -> 0.75
            source_ionic_strengths = [0.6, 0.45, 0.3]
            target_ionic_strength = 0.75
            n_iterations = 100
            n_samples_per_it = 1
        elif(args.benchmark == "batch_flory_high_to_low"):
            #High to low source -> [0.15, 0.3, 0.45] | target -> 0.01
            source_ionic_strengths = [0.15, 0.3, 0.45]
            target_ionic_strength = 0.01
            n_iterations = 10
            n_samples_per_it = 10
        elif(args.benchmark == "batch_flory_low_to_high"):
            #Low to high source -> [0.6, 0.45, 0.3] | target -> 0.75
            source_ionic_strengths = [0.6, 0.45, 0.3]
            target_ionic_strength = 0.75
            n_iterations = 10
            n_samples_per_it = 10
        else:
            raise Exception("Wrong benchmark name")
        
        n_iterations = int(args.n_sampled_points/n_samples_per_it)

        for i in tqdm(range(args.n_seeds)):
            set_seeds(i)

            source_model_list = []
            #Set up initial points
            candidate_grid = get_grid(0.001,0.999, candidate_grid_size)
            candidate_grid = grid_mask(candidate_grid)

            indices = np.random.choice(len(candidate_grid), size=n_initial_points, replace=False)
            initial_sample_points = candidate_grid[indices]

            for ionic_strength in source_ionic_strengths:
                model, _, _ = flory_source_active_learning_loop(ionic_strength, initial_sample_points =initial_sample_points, n_iterations=n_iterations,
                                                           n_samples_per_it=n_samples_per_it, candidate_size=args.candidate_size)
                source_model_list.append(model)

            print("Vanilla Model")
            print("--------------------------")
            #Save vanilla error curve
            vanilla_model, vanilla_error_curve, vanilla_sampled_points = flory_source_active_learning_loop(target_ionic_strength,initial_sample_points =initial_sample_points,
                                                                                                           n_iterations=n_iterations, n_samples_per_it=n_samples_per_it,
                                                                                                           candidate_size=args.candidate_size)
            error_curve_dict["vanilla"].append(vanilla_error_curve)

            print("Single-Source PhaseTransfer")
            print("--------------------------")
            #Save single TL error curve
            tl_single_model, tl_single_error_curve, tl_single_sampled_points = flory_tl_active_learning_loop([source_model_list[0]], target_ionic_strength, initial_sample_points =initial_sample_points,
                                                                                                             n_iterations=n_iterations, n_samples_per_it=n_samples_per_it,
                                                                                                             candidate_size=args.candidate_size)
            error_curve_dict["tl_single"].append(tl_single_error_curve)

            print("Multi-Source PhaseTransfer")
            print("--------------------------")
            #Save multi linear TL error curve
            tl_linear_model, tl_linear_error_curve, tl_linear_sampled_points = flory_tl_active_learning_loop(source_model_list, target_ionic_strength, initial_sample_points =initial_sample_points,
                                                                                                            prior_aggregation="linear",n_iterations=n_iterations,
                                                                                                            n_samples_per_it=n_samples_per_it,candidate_size=args.candidate_size)
            error_curve_dict["tl_linear"].append(tl_linear_error_curve)


    os.makedirs('synthetic_error_rate_data/gp/', exist_ok=True)
    with open('synthetic_error_rate_data/gp/'+original_benchmark_name+'.pkl', 'wb') as file: 
        pickle.dump(error_curve_dict, file)
