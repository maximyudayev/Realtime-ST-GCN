import sys
sys.path.append('/home/hassin/Documents/University/Thesis/Realtime-ST-GCN')

import onnx, numpy as np
from onnx import helper, numpy_helper
from models.utils import Graph

#####################

## CONFIG

#####################

import json

with open("data/skeletons/pku-mmd.json", 'r') as f:
    graph_config = json.load(f)

config = {
        "strategy": "spatial",
        "in_feat": 3,
        "stages": 1,
        "kernel": 9,
        "output_type": "logits",
        "latency": False,
        "is_bn": True,
        "is_bn_stats": True,
        "graph": graph_config,
        "num_classes": 52,
        "normalization": "LayerNorm",
        "rt-st-gcn": {
            "latency": False,
            "importance": True,
            "in_feat": 3,
            "buffer": 1,
            "stages": 1,
            "layers": 1,
            "kernel": 9,
            "in_ch":
                [64], 
                # [64,64,64,64,128,128,128,256,256],
            "out_ch":
                [64],
                # [64,64,64,128,128,128,256,256,256],
            "stride":
                [1],
                # [1,1,1,2,1,1,2,1,1],
            "residual":
                [1],
                # [1,1,1,1,1,1,1,1,1],
            "dropout":
                [0],
                # [0,0,0,0,0,0,0,0,0]
        }
    }

### BASIC VARIABLES ###
out_channels = config['rt-st-gcn']['out_ch'][0]
in_channels = config['rt-st-gcn']['in_ch'][0]
stride = config['rt-st-gcn']['stride'][0]
dropout = config['rt-st-gcn']['dropout'][0]
num_node = config['graph']['num_node']

#####################

## GRAPH STARTS

#####################

# Create an empty ONNX graph
graph = helper.make_graph(nodes=[], name='simple_graph', inputs=[], outputs=[])

# Add input tensors
input_shape = [1,3,1,25]  # Define input shape (batch_size=1 for simplicity)
input1_tensor = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, input_shape)
graph.input.extend([input1_tensor])
######################

### LAYER NORMALIZATION

######################

# Create a constant tensor for the shape
shape_tensor0 = helper.make_tensor(
    'shape_tensor',  # Name of the tensor
    onnx.TensorProto.INT64,  # Data type
    [2],  # Dimensions of the tensor
    [1, 75]  # Values of the tensor
)

# Create a constant tensor for the original shape
original_shape_tensor = helper.make_tensor(
    'original_shape_tensor',  # Name of the tensor
    onnx.TensorProto.INT64,  # Data type
    [4],  # Dimensions of the tensor
    [1, 3, 1, 25]  # Values of the tensor
)

# Add the shape tensor to the graph
shape_value_info = helper.make_tensor_value_info('shape_tensor0', onnx.TensorProto.INT64, [2])
graph.input.extend([shape_value_info])

# Add the shape tensor to the graph
original_shape_value_info = helper.make_tensor_value_info('original_shape_tensor', onnx.TensorProto.INT64, [4])
graph.input.extend([original_shape_value_info])

#Reshape the input for the mean operation (on axes 1&3) and keeping dimensions
input_reshaped = onnx.helper.make_node(
    'Reshape', 
    ['input', 'shape_tensor0'],
    ['x_reshaped_input'])


# Create a node for the mean operation
mean_node = helper.make_node(
    'Mean',  # The name of the operator
    ['x_reshaped_input'],  # Inputs
    ['mean'],  # Outputs
    # keepdims=1  # Keepdims parameter
)

# Create a constant tensor for the axes
axes_tensor_mean = helper.make_tensor('axes_tensor_mean', onnx.TensorProto.INT64,[1], [0])
# Add the axes tensor to the graph
mean_axes_value_info = helper.make_tensor_value_info('axes_tensor_mean', onnx.TensorProto.INT64, [1])
graph.input.extend([mean_axes_value_info])


# Create a node for the unsqueeze operation
unsqueeze_node_mean = helper.make_node(
    'Unsqueeze',
    ['mean', 'axes_tensor_mean'],
    ['mean_unsqueezed'],
)

# Then, compute the squared difference from the mean
diff = helper.make_node('Sub', ['x_reshaped_input', 'mean_unsqueezed'], ['diff'])
diff_sq = helper.make_node('Mul', ['diff', 'diff'], ['diff_sq'])
graph.node.extend([diff, diff_sq, input_reshaped])

# Compute the mean of the squared differences (which is variance)
var_node = helper.make_node('Mean', ['diff_sq'], ['var'])

# Create a node for the unsqueeze operation
unsqueeze_node_var = helper.make_node('Unsqueeze', ['var','axes_tensor_mean'], ['var_unsqueezed'])  # Specify the dimensions to be expanded


# Create a node for the square root operation
sqrt_node = helper.make_node(
    'Sqrt',  # The name of the operator
    ['var_unsqueezed'],  # Inputs
    ['std'],  # Outputs
)

# Create a node for the subtraction operation
sub_node = helper.make_node(
    'Sub',  # The name of the operator
    ['x_reshaped_input', 'mean_unsqueezed'],  # Inputs
    ['sub'],  # Outputs
)

# Create a node for the division operation
div_node = helper.make_node(
    'Div',  # The name of the operator
    ['sub', 'std'],  # Inputs
    ['div'],  # Outputs
)

reshape_back = onnx.helper.make_node(
    'Reshape', 
    ['div', 'original_shape_tensor'],
    ['reshape_back'])

norm_weight_tensor = helper.make_tensor_value_info('norm_weight_tensor', onnx.TensorProto.FLOAT, [3,1,25])
norm_bias_tensor = helper.make_tensor_value_info('norm_bias_tensor', onnx.TensorProto.FLOAT, [3,1,25])
graph.input.extend([norm_weight_tensor, norm_bias_tensor])

# Create a node for the multiplication operation
mul_node = helper.make_node(
    'Mul',  # The name of the operator
    ['reshape_back', 'norm_weight_tensor'],  # Inputs
    ['mul'],  # Outputs
)

# Create a node for the addition operation
add_node = helper.make_node(
    'Add',  # The name of the operator
    ['mul', 'norm_bias_tensor'],  # Inputs
    ['layer_norm_output'],  # Outputs
)

# Add the nodes to the graph
graph.node.extend([mean_node, var_node, sqrt_node, sub_node, div_node, mul_node, add_node, unsqueeze_node_var, unsqueeze_node_mean, reshape_back])


######################

### FCN FOR FEATURE REMAPPING

######################

# Define the weight and bias tensors
weight_tensor0 = helper.make_tensor_value_info('W0', onnx.TensorProto.FLOAT, [64, 3, 1, 1])
bias_tensor0 = helper.make_tensor_value_info('B0', onnx.TensorProto.FLOAT, [64])

# Create a node for the convolution operation
conv_node1 = helper.make_node(
    'Conv',  # The name of the operator
    ['layer_norm_output', 'W0', 'B0'],  # Inputs
    ['fcn_in_output'],  # Outputs
    kernel_shape=[1, 1],  # The size of the kernel
    pads=[0, 0, 0, 0],  # The padding for the operation
)

# Add the node to the graph
graph.node.extend([conv_node1])

# Add the tensors to the graph
graph.input.extend([weight_tensor0, bias_tensor0])

######################

### (1ST) ONLINE ST-GCN LAYER

######################

### RESIDUAL BRANCH HANDLING ###
# residual = config['rt-st-gcn']['residual'][0]

# if residual and not ((in_channels == out_channels) and (stride == 1)):
#     #THIS CASE WON'T HAPPEN NOW
#     #Define weight tensor for residual convolution:
#     weight_res_conv = helper.make_tensor_value_info('W_residual_conv', onnx.TensorProto.FLOAT, [64, 64, 1, 1])
#     # Create Conv node
#     conv_node_residual = helper.make_node(
#         'Conv',
#         inputs=['fcn_output', 'W_residual_conv'], #NO BIAS!
#         outputs=['Y'],
#         kernel_shape=[1, 1],
#         pads=[0, 0, 0, 0]
#     )




### CONVOLUTION HANDLING ###

model_graph = Graph(strategy=config['strategy'], **config['graph'])

# Let's assume that self.graph.A is a numpy array
A = np.array(model_graph.A, dtype=np.float32)
tensor_A = helper.make_tensor('A', onnx.TensorProto.FLOAT, A.shape, A.flatten().tolist()) #Need to have a consistent input size
graph.initializer.extend([tensor_A])
num_partitions = A.shape[0]

# Define the weight and bias tensors
weight_tensor1 = helper.make_tensor_value_info('W1', onnx.TensorProto.FLOAT, [config['rt-st-gcn']['out_ch'][0]*num_partitions, config['rt-st-gcn']['in_ch'][0], 1, 1])
bias_tensor1 = helper.make_tensor_value_info('B1', onnx.TensorProto.FLOAT, [config['rt-st-gcn']['out_ch'][0]*num_partitions])

# convolution of incoming frame (node-wise)
conv_node2 = helper.make_node(
    'Conv',  # The name of the operator
    ['fcn_in_output', 'W1', 'B1'],  # Inputs
    ['st-gcn_conv_out'],  # Outputs
    kernel_shape=[1, 1],  # The size of the kernel
    pads=[0, 0, 0, 0],  # The padding for the operation
)


# Add the convolution node to the graph
graph.node.extend([conv_node2])
# Add the tensors to the graph
graph.input.extend([weight_tensor1, bias_tensor1])

###  AGGREGATE-STGCN  ###

#Splitting
# Create a value for the 'split' attribute
# Create a constant tensor for the split
split_tensor = helper.make_tensor(
    'split_tensor',  # Name of the tensor
    onnx.TensorProto.INT64,  # Data type
    [1],  # Dimensions of the tensor
    [out_channels]  # Values of the tensor
)

# Add the split tensor to the graph
split_value_info = helper.make_tensor_value_info('split_tensor', onnx.TensorProto.INT64, [1])
graph.input.extend([split_value_info])

# Create a node for the Split operator
split_node = helper.make_node(
    'Split', # operator name
    ['st-gcn_conv_out', 'split_tensor'], # inputs
    ['split_o1', 'split_o2', 'split_o3'], # outputs
    axis=1, # dimension to split on
)

# Concat node
concat_node1 = helper.make_node(
    'Concat', # operator name
    ['split_o1', 'split_o2', 'split_o3'], # inputs
    ['concat1_out'], # output
    axis=-1 # dimension to concatenate on
)

# Transpose node
transpose_node = helper.make_node(
    'Transpose', # operator name
    ['concat1_out'], # input
    ['transpose_out'], # output
    perm=[0,2,4,1,3] # permutation of the dimensions
)

# MatMul node
matmul_node = helper.make_node(
    'MatMul', # operator name
    ['transpose_out', 'A'], # inputs
    ['matmul_out'] # output
)

# Create a constant tensor for the axes
axes_tensor_sum = helper.make_tensor('axes_tensor_sum', onnx.TensorProto.INT64, [1], [2])

# Add the axes tensor to the graph
axes_value_info = helper.make_tensor_value_info('axes_tensor_sum', onnx.TensorProto.INT64, [1])
graph.input.extend([axes_value_info])

# Sum node
reduce_sum_node1 = helper.make_node(
    'ReduceSum', # operator name
    ['matmul_out', 'axes_tensor_sum'], # inputs
    ['x_sum'], # output
    keepdims=0 # keep the reduced dimensions
)

kernel_size = config['kernel']
fifo_size = stride*(kernel_size-1)+1
fifo_tensor = helper.make_tensor_value_info('fifo_in', onnx.TensorProto.FLOAT, [1, fifo_size, out_channels, 25])
graph.input.extend([fifo_tensor])

# Create constant tensors for 'starts' and 'ends'
starts_tensor = helper.make_tensor('starts_tensor', onnx.TensorProto.INT64, [1], [0])
ends_tensor = helper.make_tensor('ends_tensor', onnx.TensorProto.INT64, [1], [9-1])

# Add the 'starts' and 'ends' tensors to the graph
starts_value_info = helper.make_tensor_value_info('starts_tensor', onnx.TensorProto.INT64, [1])
ends_value_info = helper.make_tensor_value_info('ends_tensor', onnx.TensorProto.INT64, [1])
graph.input.extend([starts_value_info, ends_value_info])

# Create a node for the Slice operator
slice_node1 = helper.make_node(
    'Slice', # operator name
    ['fifo_in', 'starts_tensor', 'ends_tensor'], # inputs
    ['fifo_sliced'], # outputs
)

# Create a node for the Concat operator
concat_node2 = helper.make_node(
    'Concat', # operator name
    ['x_sum', 'fifo_sliced'], # inputs
    ['fifo_concatted'], # output
    axis=1 # dimension to concatenate on
)

# Create constant tensors for 'starts', 'ends', 'axes', and 'steps'
slice2_starts_tensor = helper.make_tensor('slice2_starts_tensor', onnx.TensorProto.INT64, [1], [0])
slice2_ends_tensor = helper.make_tensor('slice2_ends_tensor', onnx.TensorProto.INT64, [1], [9])
slice2_axes_tensor = helper.make_tensor('slice2_axes_tensor', onnx.TensorProto.INT64, [1], [1])
slice2_steps_tensor = helper.make_tensor('slice2_steps_tensor', onnx.TensorProto.INT64, [1], [config['rt-st-gcn']['stride'][0]])

# Add the 'starts', 'ends', 'axes', and 'steps' tensors to the graph
slice2_starts_value_info = helper.make_tensor_value_info('slice2_starts_tensor', onnx.TensorProto.INT64, [1])
slice2_ends_value_info = helper.make_tensor_value_info('slice2_ends_tensor', onnx.TensorProto.INT64, [1])
slice2_axes_value_info = helper.make_tensor_value_info('slice2_axes_tensor', onnx.TensorProto.INT64, [1])
slice2_steps_value_info = helper.make_tensor_value_info('slice2_steps_tensor', onnx.TensorProto.INT64, [1])
graph.input.extend([slice2_starts_value_info, slice2_ends_value_info, slice2_axes_value_info, slice2_steps_value_info])

# Create a node for the Slice operator
slice_node2 = helper.make_node(
    'Slice', # operator name
    ['fifo_concatted', 'slice2_starts_tensor', 'slice2_ends_tensor', 'slice2_axes_tensor', 'slice2_steps_tensor'], # inputs
    ['a'], # outputs
)

# Create a node for the ReduceSum operator
reduce_sum_node2 = helper.make_node(
    'ReduceSum', # operator name
    ['a','axes_tensor_sum'], # inputs
    ['b'], # outputs
    # axes=[1], # dimensions to reduce
    keepdims=0 # keep the reduced dimensions
)

#USING THE SAME AXES TENSOR AS THE SUM EARLIER AS IT IS THE SAME
# Create a node for the Unsqueeze operator
unsqueeze_node = helper.make_node(
    'Unsqueeze', # operator name
    ['b', 'axes_tensor_sum'], # inputs
    ['return_aggregate_stgcn'] # outputs
    # axes=[2] # dimensions to unsqueeze
)

graph.node.extend([split_node, concat_node1, transpose_node, matmul_node, reduce_sum_node1, slice_node1, concat_node2, slice_node2])
graph.node.extend([reduce_sum_node2, unsqueeze_node])


### NORMALIZATION AND DROPOUT ON THE MAIN BRANCH ###
# Create nodes for the ReduceMean, ReduceSumSquare, Sqrt, Sub, Div, Mul, and Add operators
main_mean_node = helper.make_node(
    'ReduceMean', # operator name
    ['return_aggregate_stgcn'], # inputs
    ['main_mean'], # outputs
    axes=[1,3], # dimensions to reduce
    keepdims=1 # keep the reduced dimensions
)

reduce_sum_square_node = helper.make_node(
    'ReduceSumSquare', # operator name
    ['main_mean'], # inputs
    ['main_sum_square'], # outputs
    axes=[1,3], # dimensions to reduce
    keepdims=1 # keep the reduced dimensions
)

sqrt_node = helper.make_node(
    'Sqrt', # operator name
    ['main_sum_square'], # inputs
    ['main_std'], # outputs
)

sub_node = helper.make_node(
    'Sub', # operator name
    ['return_aggregate_stgcn', 'main_mean'], # inputs
    ['main_sub'], # outputs
)

div_node = helper.make_node(
    'Div', # operator name
    ['main_sub', 'main_std'], # inputs
    ['main_div'], # outputs
)

weight_tensor2 = helper.make_tensor_value_info('W2', onnx.TensorProto.FLOAT, [64, 1, 25])
bias_tensor2 = helper.make_tensor_value_info('B2', onnx.TensorProto.FLOAT, [64,1,25])

graph.input.extend([weight_tensor2, bias_tensor2])


mul_node = helper.make_node(
    'Mul', # operator name
    ['main_div', 'W2'], # inputs
    ['main_mul'], # outputs
)

add_node = helper.make_node(
    'Add', # operator name
    ['main_mul', 'B2'], # inputs
    ['mainbranch_layer_norm'], # outputs
)

# Create a node for the Relu operator
relu_node = helper.make_node(
    'Relu', # operator name
    ['mainbranch_layer_norm'], # inputs
    ['mainbranch_bn_relu'], # outputs
)

graph.node.extend([main_mean_node, reduce_sum_square_node, sqrt_node, sub_node, div_node, mul_node, add_node, relu_node])

### FUNCTIONAL ADD ###
# add the branches (main + residual), activate and dropout
add_node_functional = helper.make_node(
    'Add',
    inputs=['mainbranch_bn_relu', 'fcn_in_output'],
    outputs=['out_functional_add'],
)

graph.node.extend([add_node_functional])

### PERFORM DROPOUT AND RELU ###
# Create Relu node
relu_online_out_node = helper.make_node(
    'Relu',
    inputs=['out_functional_add'],
    outputs=['relu_online_out'],
)

# Create Dropout node
dropout_node = helper.make_node(
    'Dropout',
    inputs=['relu_online_out'],
    outputs=['dropout_out'],
    ratio=dropout,
)

graph.node.extend([relu_online_out_node, dropout_node])


######################

### AVG POOLING LAYER ###

######################
# Create AveragePool node

avg_pool_node = helper.make_node(
    'AveragePool',
    inputs=['dropout_out'],
    outputs=['avg_pool_out'],
    kernel_shape=[1, num_node],
    strides=[1, 1],
    pads=[0, 0, 0, 0]
)

graph.node.extend([avg_pool_node])

######################

### FCN_OUT LAYER ###

######################
# Define the weight and bias tensors
weight_tensor3 = helper.make_tensor_value_info('W3', onnx.TensorProto.FLOAT, [52, 64, 1, 1])
bias_tensor3 = helper.make_tensor_value_info('B3', onnx.TensorProto.FLOAT, [52])

# Create a node for the convolution operation
conv_node_end = helper.make_node(
    'Conv',  # The name of the operator
    ['avg_pool_out', 'W3', 'B3'],  # Inputs
    ['fcn_out_output'],  # Outputs
    kernel_shape=[1, 1],  # The size of the kernel
    pads=[0, 0, 0, 0],  # The padding for the operation
)

# Add the node to the graph
graph.node.extend([conv_node_end])
graph.input.extend([weight_tensor3, bias_tensor3])

######################

### OUTPUT SLICING YAYYYY ###

######################
# Create Slice node
slice_node_out = helper.make_node(
    'Slice',
    inputs=['fcn_out_output'],
    outputs=['model_output'],
    axes=[3],
    starts=[0],
    ends=[1]
)

graph.node.extend([slice_node_out])

# Add output tensor
output_tensor = helper.make_tensor_value_info('model_output', onnx.TensorProto.FLOAT, [1, 52, 1])
graph.output.extend([output_tensor])

# Create the model with the graph
model = helper.make_model(graph)

# Compile and validate the model
# onnx.checker.check_model(model)

# Export the model to an ONNX file
onnx.save(model, 'custom.onnx')
