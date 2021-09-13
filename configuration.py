import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--centers_initial_range", default=[0,1], help="numerical range of centers")
parser.add_argument("--num_centers", type=int, default=10, help="number of centers for soft quantization")
parser.add_argument("--regularization_factor_centers", type=int, default=0, help="regularization factor of centers")


## test for validity

# opt = parser.parse_args()
# print(opt.centers_initial_range)