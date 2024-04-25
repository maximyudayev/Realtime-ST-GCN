import torch
import onnx
import numpy as np

# 1. Define a simple computational graph (PyTorch-like)
x = torch.randn(1, 3, 224, 224, requires_grad=False)
y = x + 2
z = torch.relu(y)

# 2. Create an empty ONNX model
onnx_model = onnx.ModelProto()
onnx_model.graph.name = "simple_onnx_example" 

# 3. Define inputs and outputs
onnx_model.graph.input.add().CopyFrom(onnx.helper.make_tensor_value_info('input_x', onnx.TensorProto.FLOAT, [1, 3, 224, 224]))
onnx_model.graph.output.add().CopyFrom(onnx.helper.make_tensor_value_info('output_z', onnx.TensorProto.FLOAT, [1, 3, 224, 224]))

# 4. Build the graph (add nodes for operations)

# 4.1 Batch normalization

onnx_model.graph.node.add().CopyFrom(onnx.helper.make_node('Add', ['input_x', '2'], ['y'], name='add_node'))
onnx_model.graph.node.add().CopyFrom(onnx.helper.make_node('Relu', ['y'], ['output_z'], name='relu_node'))

# 5. Modified line with an empty shape list
onnx_tensor = onnx.helper.make_tensor('2', onnx.TensorProto.FLOAT, (), [2]) 
onnx_model.graph.initializer.add().CopyFrom(onnx_tensor)

# 6. Save the ONNX model
onnx.save(onnx_model, 'simple_model.onnx')
