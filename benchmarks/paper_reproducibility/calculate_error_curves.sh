#This script runs the code for calculating the GP related error curves

#Sine Wave Benchmark
python gp_error_curve_reproducer.py --benchmark sine --n_seeds 50 --n_sampled_points 100 --n_initial_points 10
python gp_error_curve_reproducer.py --benchmark noisy_sine --n_seeds 50 --n_sampled_points 100 --n_initial_points 10
python gp_error_curve_reproducer.py --benchmark batch_sine --n_seeds 50 --n_sampled_points 100 --n_initial_points 10

#3-component Biological Condensate Benchmark
python gp_error_curve_reproducer.py --benchmark biological_condensate --n_seeds 50 --n_sampled_points 50 --n_initial_points 10
python gp_error_curve_reproducer.py --benchmark batch_biological_condensate --n_seeds 50 --n_sampled_points 50 --n_initial_points 10

#Supramolecular Copolymerization Benchmark
python gp_error_curve_reproducer.py --benchmark supramolecular_copolymerization --n_seeds 50 --n_sampled_points 50 --n_initial_points 5
python gp_error_curve_reproducer.py --benchmark batch_supramolecular_copolymerization --n_seeds 50 --n_sampled_points 50 --n_initial_points 5


#This script runs the code for calculating the PDC related error curves
#Sine Wave Benchmark
python pdc_error_curve_reproducer.py --benchmark sine --n_seeds 50 --n_sampled_points 100 --n_initial_points 10
python pdc_error_curve_reproducer.py --benchmark noisy_sine --n_seeds 50 --n_sampled_points 100 --n_initial_points 10

#3-component Biological Condensate Benchmark
python pdc_error_curve_reproducer.py --benchmark biological_condensate --n_seeds 50 --n_sampled_points 50 --n_initial_points 10

#Supramolecular Copolymerization Benchmark
python pdc_error_curve_reproducer.py --benchmark supramolecular_copolymerization --n_seeds 50 --n_sampled_points 50 --n_initial_points 5