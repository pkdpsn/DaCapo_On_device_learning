layer_costs = [10, 15, 20, 25, 15, 10, 20, 25, 15, 30, 10, 20] 
available_memory = 100  

def computation_cost(segment, layer_costs):
    cost = 0
    for i in range(len(segment)):
        cost += layer_costs[segment[i]] * (len(segment) - i)
    return cost

def max_segment_mem_req(segment, memory_requirements):
    """Calculate the total memory requirement of a segment."""
    return sum(memory_requirements[i] for i in segment)

def optimal_seg_finder(cur_seg, index, n, M, min_cost, optimal_seg, memory_requirements):
    """
    Recursively find the optimal segmentation that meets memory constraints and minimizes cost.
    cur_seg: current segment being built
    index: current layer index
    n: total number of layers
    M: available memory size
    min_cost: minimum computation cost found so far
    optimal_seg: stores the optimal segmentation
    memory_requirements: list of memory requirements per layer
    """
    if cur_seg and max_segment_mem_req(cur_seg, memory_requirements) <= M:
        cost = computation_cost(cur_seg, layer_costs)
        if cost < min_cost[0]:
            min_cost[0] = cost
            optimal_seg[:] = cur_seg[:]

    for i in range(index, n):
        # Only continue if adding this layer keeps the memory requirement within bounds
        if max_segment_mem_req(cur_seg + [i], memory_requirements) <= M:
            optimal_seg_finder(cur_seg + [i], i + 1, n, M, min_cost, optimal_seg, memory_requirements)

# Memory requirements for each layer in KB
memory_requirements = [10, 20, 15, 25, 10, 10, 20, 25, 15, 10, 20, 30]

# Variables to store the results
min_cost = [float('inf')]
optimal_seg = []

# Run the optimal segmentation finder for 12 layers
optimal_seg_finder([], 0, 12, available_memory, min_cost, optimal_seg, memory_requirements)

print("Optimal Segmentation:", optimal_seg)
print("Minimum Computation Cost:", min_cost[0])

# Rank Finder as per the algorithm in the paper
def rank_finder(tk, mk, M):
    consumed = 0
    ranks = [0] * tk
    current_rank = 1
    p = tk - 1

    while p >= 0:
        if consumed + mk[p] > M:
            consumed = mk[p]
            current_rank += 1
        else:
            consumed += mk[p]
        ranks[p] = current_rank
        p -= 1

    return ranks

# Define the segmentation scheme (e.g., [3, 7, 9, 12] as given in the paper)
segmentation_scheme = [3, 7, 9, 12]
mk = [memory_requirements[i - 1] for i in segmentation_scheme]

# Find the ranks for each segment
ranks = rank_finder(len(segmentation_scheme), mk, available_memory)

print("Segment Ranks:", ranks)
