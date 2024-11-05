import torch.nn as nn
device = 'cpu'


def compute_cost(layer, input_shape):
    """
    Compute the computation cost C_i for a given layer.

    Parameters:
    - layer: The layer to compute the cost for.
    - input_shape: Tuple representing the shape (batch_size, channels, height, width) for conv layers,
                   (batch_size, features) for dense layers.

    Returns:
    - cost: Computation cost C_i for the layer.
    """
    if isinstance(layer, nn.Conv2d):
        output_channels = layer.out_channels
        input_channels = layer.in_channels
        kernel_height, kernel_width = layer.kernel_size
        output_height, output_width = input_shape[2], input_shape[3]

        cost = (output_height * output_width * output_channels *
                kernel_height * kernel_width * input_channels)

    elif isinstance(layer, nn.Linear):
        input_features = layer.in_features
        output_features = layer.out_features
        cost = input_features * output_features
    else:
        cost = 0
    return cost


def compute_layer_costs(model, input_shape):
    """
    Compute and return the computation cost C_i for each layer in the model.

    Parameters:
    - model: The neural network model.
    - input_shape: The input shape for the model.

    Returns:
    - layer_costs: A list of tuples, each containing the layer name and its computation cost.
    """
    layer_costs = []
    layer_num = 1  # Start layer numbering at 1

    for layer in model.children():
        if isinstance(layer, nn.Sequential):
            for sublayer in layer:
                input_shape = (
                input_shape[0], layer[0].out_channels, input_shape[2], input_shape[3])  # Update input shape
                layer_cost = compute_cost(sublayer, input_shape)
                if layer_cost == 0:
                    continue
                layer_costs.append(layer_cost)
                layer_num += 1
        else:
            layer_cost = compute_cost(layer, input_shape)
            if layer_cost == 0:
                continue
            layer_costs.append(layer_cost)
            if isinstance(layer, nn.Linear):
                input_shape = (input_shape[0], layer.out_features)  # Update for next layer if dense layer
            layer_num += 1

    return layer_costs


def layer_memory_req(layer, input_shape):
    """
    Calculate memory requirement for a single layer based on its type and parameters.

    Parameters:
    - layer: The layer object (either Conv2d or Linear).
    - input_shape: Tuple representing the shape (batch_size, channels, height, width) for conv layers,
                   (batch_size, features) for dense layers.

    Returns:
    - memory_req_kb: Memory requirement for the layer in kilobytes (KB).
    """
    if isinstance(layer, nn.Conv2d):
        # Conv2D layer parameters
        out_channels = layer.out_channels
        kernel_size = layer.kernel_size[0]
        in_channels = layer.in_channels
        output_height, output_width = input_shape[2], input_shape[3]

        # Memory for activations (output size) + parameters (weights)
        activations_mem = output_height * output_width * out_channels * 4  # 4 bytes per float
        weights_mem = (kernel_size * kernel_size * in_channels * out_channels) * 4  # Weights
        memory_req = activations_mem + weights_mem

    elif isinstance(layer, nn.Linear):
        # Linear layer parameters
        in_features = layer.in_features
        out_features = layer.out_features

        # Memory for activations (output size) + parameters (weights)
        activations_mem = out_features * 4  # 4 bytes per float
        weights_mem = in_features * out_features * 4  # Weights
        memory_req = activations_mem + weights_mem

    else:
        memory_req = 0  # Unsupported layer types

    # Convert bytes to kilobytes
    memory_req_kb = memory_req / 1024
    return memory_req_kb


def layer_memory_list(model, input_shape):
    """
    Calculate the memory requirement for each layer in the model and return a list.

    Parameters:
    - model: The neural network model.
    - input_shape: The input shape for the model.

    Returns:
    - layer_memory_list: List of memory requirements for each layer in kilobytes (KB).
    """
    layer_memory_list = []
    layer_num = 1  # Start layer numbering at 1

    # Iterate over the model layers
    for layer in model.children():
        if isinstance(layer, nn.Sequential):
            for sublayer in layer:
                layer_memory = layer_memory_req(sublayer, input_shape)
                if layer_memory != 0:
                    layer_memory_list.append(layer_memory)
                    # print(f"Layer {layer_num}: Memory Requirement = {layer_memory} KB")
                    layer_num += 1
        else:
            layer_memory = layer_memory_req(layer, input_shape)
            if layer_memory != 0:
                layer_memory_list.append(layer_memory / 4)
                # print(f"Layer {layer_num}: Memory Requirement = {layer_memory} KB")
                layer_num += 1

    return layer_memory_list


def Rkp(Ci, F, L):
    """
    Calculate the weighted computation cost for a segment of layers.

    Parameters:
    - Ci: List of computation costs for each layer.
    - F: Starting index of the segment (exclusive).
    - L: Ending index of the segment (inclusive).

    Returns:
    - sum: Total weighted computation cost for the segment.
    """
    sum = 0
    if F == 0:
        for i in range(F, L):
            # print(i, i - F+1)
            sum += Ci[i] * (i - F + 1)
    else:
        for i in range(F + 1, L):
            # print(i, i - F)
            sum += Ci[i] * (i - F)
    return sum


def computation_cost(Ci, seg):
    """
    Calculate the total computation cost across multiple segments of layers.

    Parameters:
    - Ci: List of computation costs for each layer.
    - segment: List of segment boundaries (layer indices). Each boundary marks the end of a segment.

    Returns:
    - sum: Total computation cost across all segments.
    """
    sum = 0
    prev = 0
    segment = seg.copy()
    segment.append(len(Ci) - 1)  # Ensure the last segment goes to the end of Ci

    for i in range(len(segment)):
        # print(prev, segment[i])  # Debug: Print segment start and end indices
        sum += Rkp(Ci, prev, segment[i])
        prev = segment[i]

    return sum

def max_mem_req(Mi , curr_seg):
    mem_sum = 0
    for i in range(len(curr_seg)):
        mem_sum += Mi[curr_seg[i]]
    return mem_sum


