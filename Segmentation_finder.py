import torch.nn as nn
from Seg_utils import compute_layer_costs , layer_memory_list , Rkp , computation_cost , max_mem_req
from models_Tiny_Vgg.model3C_2F import MNIST_CNN
import time
device = 'cpu'


def optimalSegFinder(Ci, Mi, M, n):
    """
    Finds the optimal segmentation of layers with minimal computation cost, subject to memory constraints.

    Parameters:
    - Ci: List of computation costs for each layer.
    - M: Available memory size.
    - n: Total number of layers.

    Returns:
    - optimal_seg: Optimal segmentation that minimizes computation cost.
    - min_cost: Minimum computation cost of the optimal segmentation.
    """
    # Initialize variables
    min_cost = float('inf')
    optimal_seg = []
    cur_seg = []

    def recursive_optimal_seg_finder(cur_seg, index):
        nonlocal min_cost, optimal_seg

        # Check if the current segment meets memory requirements
        if max_mem_req(Mi, cur_seg) <= M:
            # Calculate the computation cost for the current segmentation
            cost = computation_cost(Ci, cur_seg)
            # print(f" {cost} {cur_seg} ")
            # Update minimum cost and optimal segmentation if a better solution is found
            if cost < min_cost:
                min_cost = cost
                optimal_seg = cur_seg[:]

        # Iterate over possible next layers to add to the current segmentation
        for i in range(index, n):
            # Recursively find the optimal segmentation with the next layer added to cur_seg
            # print(f"adding {i} to curr {cur_seg}")
            recursive_optimal_seg_finder(cur_seg + [i], i + 1)

    # Start the recursion with an empty segmentation and at the first layer
    recursive_optimal_seg_finder(cur_seg, 1)

    return optimal_seg, min_cost
def main():
    input_shape = (1, 1, 28, 28)
    model = MNIST_CNN(input_shape=1, hidden_units=10, output_shape=10).to(device)
    Ci = compute_layer_costs(model, input_shape)
    Mi = layer_memory_list(model, input_shape)
    M = 200
    n = len(Ci) - 1
    start = time.time()
    optimal_seg, min_cost = optimalSegFinder(Ci, Mi, M, n)
    end = time.time()
    print("Optimal Segmentation:", optimal_seg)
    print("Minimum Computation Cost:", min_cost)
    print("Time taken:", end - start)


if __name__ == '__main__':
    main()